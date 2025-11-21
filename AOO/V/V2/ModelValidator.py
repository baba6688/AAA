"""
V2模型验证器
===========

这是一个全面的模型验证器，用于验证机器学习模型的各种属性和性能指标。
该验证器提供了数据验证、模型结构验证、参数验证、兼容性检查等功能。

主要功能：
1. 数据验证和清洗 - 验证输入数据的质量和完整性
2. 模型结构验证 - 验证模型架构的合理性
3. 参数验证和边界检查 - 验证模型参数的合法性
4. 模型兼容性检查 - 验证模型与环境的兼容性
5. 数据集分割和验证 - 智能的数据集分割策略
6. 交叉验证实现 - 多种交叉验证方法
7. 验证结果统计 - 详细的验证结果分析
8. 验证报告生成 - 专业的验证报告输出
9. 验证错误处理 - 完善的错误处理机制

作者: 智能量化系统开发团队
版本: 2.0.0
日期: 2025-11-05
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
import logging
import json
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, 
    cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, mean_squared_error,
    mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.exceptions import NotFittedError
import joblib


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    验证结果数据类
    
    存储模型验证的完整结果，包括各种指标、错误信息和统计信息。
    """
    # 基本信息
    model_name: str
    validation_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 验证状态
    is_valid: bool = True
    validation_score: float = 0.0
    
    # 性能指标
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    
    # 交叉验证结果
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # 数据验证结果
    data_quality_score: float = 1.0
    missing_values_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    outlier_ratio: float = 0.0
    
    # 模型结构验证
    structure_valid: bool = True
    parameter_count: int = 0
    complexity_score: float = 0.0
    
    # 错误和警告
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info_messages: List[str] = field(default_factory=list)
    
    # 验证详情
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """将验证结果转换为字典格式"""
        return {
            'model_name': self.model_name,
            'validation_type': self.validation_type,
            'timestamp': self.timestamp,
            'is_valid': self.is_valid,
            'validation_score': self.validation_score,
            'performance_metrics': {
                'accuracy': self.accuracy,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1_score,
                'roc_auc': self.roc_auc,
                'mse': self.mse,
                'mae': self.mae,
                'r2_score': self.r2_score
            },
            'cross_validation': {
                'cv_scores': self.cv_scores,
                'cv_mean': self.cv_mean,
                'cv_std': self.cv_std
            },
            'data_quality': {
                'quality_score': self.data_quality_score,
                'missing_values_ratio': self.missing_values_ratio,
                'duplicate_ratio': self.duplicate_ratio,
                'outlier_ratio': self.outlier_ratio
            },
            'model_structure': {
                'structure_valid': self.structure_valid,
                'parameter_count': self.parameter_count,
                'complexity_score': self.complexity_score
            },
            'validation_output': {
                'errors': self.errors,
                'warnings': self.warnings,
                'info_messages': self.info_messages
            },
            'details': self.validation_details
        }
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """将验证结果保存为JSON格式"""
        result_dict = self.to_dict()
        json_str = json.dumps(result_dict, indent=2, ensure_ascii=False)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"验证结果已保存到: {filepath}")
        
        return json_str
    
    def generate_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("=" * 60)
        report.append(f"模型验证报告 - {self.model_name}")
        report.append("=" * 60)
        report.append(f"验证类型: {self.validation_type}")
        report.append(f"验证时间: {self.timestamp}")
        report.append(f"验证状态: {'通过' if self.is_valid else '失败'}")
        report.append(f"验证得分: {self.validation_score:.4f}")
        report.append("")
        
        # 性能指标
        if any([self.accuracy, self.precision, self.recall, self.f1_score]):
            report.append("性能指标:")
            if self.accuracy is not None:
                report.append(f"  准确率: {self.accuracy:.4f}")
            if self.precision is not None:
                report.append(f"  精确率: {self.precision:.4f}")
            if self.recall is not None:
                report.append(f"  召回率: {self.recall:.4f}")
            if self.f1_score is not None:
                report.append(f"  F1得分: {self.f1_score:.4f}")
            if self.roc_auc is not None:
                report.append(f"  ROC-AUC: {self.roc_auc:.4f}")
            report.append("")
        
        # 回归指标
        if any([self.mse, self.mae, self.r2_score]):
            report.append("回归指标:")
            if self.mse is not None:
                report.append(f"  均方误差: {self.mse:.4f}")
            if self.mae is not None:
                report.append(f"  平均绝对误差: {self.mae:.4f}")
            if self.r2_score is not None:
                report.append(f"  R²得分: {self.r2_score:.4f}")
            report.append("")
        
        # 交叉验证结果
        if self.cv_scores:
            report.append("交叉验证结果:")
            report.append(f"  平均得分: {self.cv_mean:.4f}")
            report.append(f"  标准差: {self.cv_std:.4f}")
            report.append(f"  得分范围: [{min(self.cv_scores):.4f}, {max(self.cv_scores):.4f}]")
            report.append("")
        
        # 数据质量
        report.append("数据质量:")
        report.append(f"  质量得分: {self.data_quality_score:.4f}")
        report.append(f"  缺失值比例: {self.missing_values_ratio:.4f}")
        report.append(f"  重复值比例: {self.duplicate_ratio:.4f}")
        report.append(f"  异常值比例: {self.outlier_ratio:.4f}")
        report.append("")
        
        # 模型结构
        report.append("模型结构:")
        report.append(f"  结构有效性: {'是' if self.structure_valid else '否'}")
        report.append(f"  参数数量: {self.parameter_count}")
        report.append(f"  复杂度得分: {self.complexity_score:.4f}")
        report.append("")
        
        # 错误和警告
        if self.errors:
            report.append("错误:")
            for error in self.errors:
                report.append(f"  ❌ {error}")
            report.append("")
        
        if self.warnings:
            report.append("警告:")
            for warning in self.warnings:
                report.append(f"  ⚠️ {warning}")
            report.append("")
        
        if self.info_messages:
            report.append("信息:")
            for info in self.info_messages:
                report.append(f"  ℹ️ {info}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


class ValidationError(Exception):
    """验证错误异常类"""
    pass


class DataValidator:
    """
    数据验证器
    
    负责数据的质量检查、清洗和预处理。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据验证器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.quality_threshold = self.config.get('quality_threshold', 0.8)
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.missing_threshold = self.config.get('missing_threshold', 0.1)
        
    def validate_data_quality(self, X: Union[pd.DataFrame, np.ndarray], 
                            y: Optional[Union[pd.Series, np.ndarray]] = None) -> Dict[str, Any]:
        """
        验证数据质量
        
        Args:
            X: 特征数据
            y: 目标数据（可选）
            
        Returns:
            数据质量检查结果
        """
        results = {
            'is_valid': True,
            'quality_score': 1.0,
            'missing_values_ratio': 0.0,
            'duplicate_ratio': 0.0,
            'outlier_ratio': 0.0,
            'data_shape': None,
            'feature_types': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # 转换为DataFrame以便处理
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X)
            else:
                X_df = X.copy()
            
            results['data_shape'] = X_df.shape
            
            # 检查缺失值
            missing_ratio = X_df.isnull().sum().sum() / (X_df.shape[0] * X_df.shape[1])
            results['missing_values_ratio'] = missing_ratio
            
            if missing_ratio > self.missing_threshold:
                results['warnings'].append(f"缺失值比例过高: {missing_ratio:.4f}")
                results['quality_score'] -= 0.2
            
            # 检查重复行
            if isinstance(X, pd.DataFrame):
                duplicate_ratio = X_df.duplicated().sum() / len(X_df)
            else:
                # 对于numpy数组，需要特殊处理
                duplicate_ratio = 0.0
                try:
                    _, unique_indices = np.unique(X, axis=0, return_index=True)
                    duplicate_ratio = 1 - len(unique_indices) / len(X)
                except:
                    pass
            
            results['duplicate_ratio'] = duplicate_ratio
            
            if duplicate_ratio > 0.1:
                results['warnings'].append(f"重复行比例过高: {duplicate_ratio:.4f}")
                results['quality_score'] -= 0.1
            
            # 检查异常值（使用IQR方法）
            outlier_ratios = []
            for col in X_df.select_dtypes(include=[np.number]).columns:
                Q1 = X_df[col].quantile(0.25)
                Q3 = X_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((X_df[col] < lower_bound) | (X_df[col] > upper_bound)).sum()
                outlier_ratios.append(outliers / len(X_df))
            
            if outlier_ratios:
                results['outlier_ratio'] = np.mean(outlier_ratios)
                if results['outlier_ratio'] > 0.05:
                    results['warnings'].append(f"异常值比例较高: {results['outlier_ratio']:.4f}")
            
            # 检查数据类型
            for col in X_df.columns:
                if X_df[col].dtype in ['object', 'category']:
                    results['feature_types'][col] = 'categorical'
                elif X_df[col].dtype in ['int64', 'float64']:
                    results['feature_types'][col] = 'numerical'
                else:
                    results['feature_types'][col] = str(X_df[col].dtype)
            
            # 检查目标变量
            if y is not None:
                if isinstance(y, np.ndarray):
                    y_series = pd.Series(y)
                else:
                    y_series = y
                
                # 检查目标变量缺失值
                y_missing = y_series.isnull().sum() / len(y_series)
                if y_missing > 0:
                    results['errors'].append(f"目标变量缺失值比例: {y_missing:.4f}")
                    results['is_valid'] = False
                
                # 检查目标变量分布
                if y_series.dtype in ['object', 'category']:
                    unique_ratio = y_series.nunique() / len(y_series)
                    if unique_ratio > 0.5:
                        results['warnings'].append("类别变量类别过多，可能影响模型性能")
            
            # 最终质量评估
            if results['quality_score'] < self.quality_threshold:
                results['is_valid'] = False
            
            logger.info(f"数据质量验证完成，质量得分: {results['quality_score']:.4f}")
            
        except Exception as e:
            results['errors'].append(f"数据质量验证失败: {str(e)}")
            results['is_valid'] = False
            logger.error(f"数据质量验证错误: {str(e)}")
        
        return results
    
    def clean_data(self, X: Union[pd.DataFrame, np.ndarray], 
                  y: Optional[Union[pd.Series, np.ndarray]] = None,
                  strategy: str = 'auto') -> Tuple[Union[pd.DataFrame, np.ndarray], 
                                                 Optional[Union[pd.Series, np.ndarray]]]:
        """
        数据清洗
        
        Args:
            X: 特征数据
            y: 目标数据（可选）
            strategy: 清洗策略 ('drop', 'fill', 'auto')
            
        Returns:
            清洗后的数据
        """
        try:
            if isinstance(X, np.ndarray):
                X_clean = pd.DataFrame(X)
            else:
                X_clean = X.copy()
            
            # 处理缺失值
            if strategy in ['auto', 'fill']:
                # 数值型变量用均值填充
                numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
                X_clean[numeric_cols] = X_clean[numeric_cols].fillna(X_clean[numeric_cols].mean())
                
                # 类别型变量用众数填充
                categorical_cols = X_clean.select_dtypes(include=['object', 'category']).columns
                X_clean[categorical_cols] = X_clean[categorical_cols].fillna(
                    X_clean[categorical_cols].mode().iloc[0] if not X_clean[categorical_cols].mode().empty else 'Unknown'
                )
            
            if strategy == 'drop':
                # 删除包含缺失值的行
                X_clean = X_clean.dropna()
                if y is not None:
                    if isinstance(y, pd.Series):
                        y_clean = y.loc[X_clean.index]
                    else:
                        # 对于numpy数组，需要重新索引
                        valid_indices = X_clean.index
                        y_clean = y[valid_indices] if hasattr(y, '__getitem__') else y
                else:
                    y_clean = None
            else:
                y_clean = y
            
            # 转换回原始格式
            if isinstance(X, np.ndarray):
                X_clean = X_clean.values
            
            logger.info(f"数据清洗完成，原始形状: {X.shape}, 清洗后形状: {X_clean.shape}")
            
            return X_clean, y_clean
            
        except Exception as e:
            logger.error(f"数据清洗失败: {str(e)}")
            raise ValidationError(f"数据清洗失败: {str(e)}")


class ModelStructureValidator:
    """
    模型结构验证器
    
    负责验证模型的架构合理性、参数合法性等。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化模型结构验证器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.max_parameters = self.config.get('max_parameters', 1000000)
        self.complexity_threshold = self.config.get('complexity_threshold', 10.0)
    
    def validate_model_structure(self, model: Any, X_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        验证模型结构
        
        Args:
            model: 待验证的模型
            X_shape: 输入数据形状
            
        Returns:
            模型结构验证结果
        """
        results = {
            'is_valid': True,
            'structure_valid': True,
            'parameter_count': 0,
            'complexity_score': 0.0,
            'compatibility_score': 1.0,
            'warnings': [],
            'errors': [],
            'details': {}
        }
        
        try:
            # 检查模型类型和基本属性
            model_type = type(model).__name__
            results['details']['model_type'] = model_type
            
            # 尝试获取模型参数数量
            try:
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    results['parameter_count'] = len(params)
                elif hasattr(model, 'n_features_in_'):
                    results['details']['n_features'] = model.n_features_in_
                elif hasattr(model, 'coef_'):
                    if hasattr(model.coef_, 'shape'):
                        results['parameter_count'] = np.prod(model.coef_.shape)
                    else:
                        results['parameter_count'] = len(model.coef_)
                elif hasattr(model, 'feature_importances_'):
                    results['parameter_count'] = len(model.feature_importances_)
                
                # 计算复杂度得分
                if results['parameter_count'] > 0:
                    results['complexity_score'] = np.log10(results['parameter_count'] + 1)
                
            except Exception as e:
                results['warnings'].append(f"无法获取模型参数信息: {str(e)}")
            
            # 检查参数数量限制
            if results['parameter_count'] > self.max_parameters:
                results['errors'].append(f"模型参数数量过多: {results['parameter_count']}")
                results['is_valid'] = False
            
            # 检查复杂度
            if results['complexity_score'] > self.complexity_threshold:
                results['warnings'].append(f"模型复杂度较高: {results['complexity_score']:.2f}")
            
            # 检查模型是否已训练
            try:
                if hasattr(model, 'fit') and hasattr(model, 'predict'):
                    # 检查是否已训练
                    if hasattr(model, 'classes_'):
                        results['details']['n_classes'] = len(model.classes_)
                    elif hasattr(model, 'n_classes_'):
                        results['details']['n_classes'] = model.n_classes_
                    
                    # 检查输入兼容性
                    if hasattr(model, 'n_features_in_'):
                        expected_features = model.n_features_in_
                        actual_features = X_shape[1] if len(X_shape) > 1 else 1
                        if expected_features != actual_features:
                            results['errors'].append(
                                f"特征数量不匹配: 期望{expected_features}, 实际{actual_features}"
                            )
                            results['is_valid'] = False
                            results['compatibility_score'] = 0.0
                        else:
                            results['compatibility_score'] = 1.0
                    
                else:
                    results['errors'].append("模型缺少必要的fit和predict方法")
                    results['is_valid'] = False
            
            except Exception as e:
                results['warnings'].append(f"模型兼容性检查失败: {str(e)}")
            
            # 特殊检查
            if 'sklearn' in str(type(model)):
                results['details']['framework'] = 'scikit-learn'
            elif 'tensorflow' in str(type(model)) or 'keras' in str(type(model)):
                results['details']['framework'] = 'tensorflow/keras'
            elif 'torch' in str(type(model)) or 'pytorch' in str(type(model)):
                results['details']['framework'] = 'pytorch'
            else:
                results['details']['framework'] = 'unknown'
            
            logger.info(f"模型结构验证完成: {model_type}, 参数数量: {results['parameter_count']}")
            
        except Exception as e:
            results['errors'].append(f"模型结构验证失败: {str(e)}")
            results['is_valid'] = False
            logger.error(f"模型结构验证错误: {str(e)}")
        
        return results


class CrossValidator:
    """
    交叉验证器
    
    实现多种交叉验证方法，包括K折、留一法、分层抽样等。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化交叉验证器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.cv_folds = self.config.get('cv_folds', 5)
        self.random_state = self.config.get('random_state', 42)
        self.scoring = self.config.get('scoring', 'accuracy')
    
    def cross_validate(self, model: Any, X: Union[pd.DataFrame, np.ndarray], 
                      y: Union[pd.Series, np.ndarray], 
                      cv_method: str = 'kfold') -> Dict[str, Any]:
        """
        执行交叉验证
        
        Args:
            model: 待验证的模型
            X: 特征数据
            y: 目标数据
            cv_method: 交叉验证方法 ('kfold', 'stratified', 'loo')
            
        Returns:
            交叉验证结果
        """
        results = {
            'cv_scores': [],
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'cv_details': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # 准备数据
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            
            if isinstance(y, pd.Series):
                y_array = y.values
            else:
                y_array = y
            
            # 选择交叉验证方法
            if cv_method == 'kfold':
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            elif cv_method == 'stratified':
                # 检查是否为分类问题
                if len(np.unique(y_array)) < len(y_array) * 0.5:  # 类别数少于样本数的50%
                    cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                else:
                    cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                    results['warnings'].append("使用K折交叉验证（数据可能不适合分层抽样）")
            elif cv_method == 'loo':
                from sklearn.model_selection import LeaveOneOut
                cv = LeaveOneOut()
                results['warnings'].append("使用留一法交叉验证，计算时间可能较长")
            else:
                raise ValueError(f"不支持的交叉验证方法: {cv_method}")
            
            # 执行交叉验证
            scores = cross_val_score(model, X_array, y_array, cv=cv, scoring=self.scoring)
            
            results['cv_scores'] = scores.tolist()
            results['cv_mean'] = float(np.mean(scores))
            results['cv_std'] = float(np.std(scores))
            results['cv_details'] = {
                'cv_method': cv_method,
                'n_splits': cv.get_n_splits(),
                'scoring': self.scoring,
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'score_range': float(np.max(scores) - np.min(scores))
            }
            
            logger.info(f"交叉验证完成: {cv_method}, 平均得分: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
            
        except Exception as e:
            results['errors'].append(f"交叉验证失败: {str(e)}")
            logger.error(f"交叉验证错误: {str(e)}")
        
        return results


class ValidationReporter:
    """
    验证报告生成器
    
    负责生成各种格式的验证报告，包括文本、JSON、图表等。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化验证报告生成器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.output_dir = Path(self.config.get('output_dir', './validation_reports'))
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_visual_report(self, validation_result: ValidationResult, 
                             save_path: Optional[str] = None) -> str:
        """
        生成可视化验证报告
        
        Args:
            validation_result: 验证结果
            save_path: 保存路径
            
        Returns:
            报告文件路径
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.output_dir / f"validation_report_{validation_result.model_name}_{timestamp}.png"
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'模型验证报告 - {validation_result.model_name}', fontsize=16, fontweight='bold')
            
            # 1. 性能指标雷达图
            ax1 = axes[0, 0]
            metrics = []
            values = []
            
            if validation_result.accuracy is not None:
                metrics.append('准确率')
                values.append(validation_result.accuracy)
            if validation_result.precision is not None:
                metrics.append('精确率')
                values.append(validation_result.precision)
            if validation_result.recall is not None:
                metrics.append('召回率')
                values.append(validation_result.recall)
            if validation_result.f1_score is not None:
                metrics.append('F1得分')
                values.append(validation_result.f1_score)
            
            if metrics and values:
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                values += values[:1]  # 闭合图形
                angles += angles[:1]
                
                ax1.plot(angles, values, 'o-', linewidth=2, color='blue')
                ax1.fill(angles, values, alpha=0.25, color='blue')
                ax1.set_xticks(angles[:-1])
                ax1.set_xticklabels(metrics)
                ax1.set_ylim(0, 1)
                ax1.set_title('性能指标雷达图')
                ax1.grid(True)
            
            # 2. 交叉验证得分分布
            ax2 = axes[0, 1]
            if validation_result.cv_scores:
                ax2.hist(validation_result.cv_scores, bins=min(10, len(validation_result.cv_scores)), 
                        alpha=0.7, color='green', edgecolor='black')
                ax2.axvline(validation_result.cv_mean, color='red', linestyle='--', 
                           label=f'均值: {validation_result.cv_mean:.4f}')
                ax2.set_xlabel('验证得分')
                ax2.set_ylabel('频次')
                ax2.set_title('交叉验证得分分布')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # 3. 数据质量指标
            ax3 = axes[1, 0]
            quality_metrics = ['质量得分', '缺失值比率', '重复值比率', '异常值比率']
            quality_values = [
                validation_result.data_quality_score,
                validation_result.missing_values_ratio,
                validation_result.duplicate_ratio,
                validation_result.outlier_ratio
            ]
            colors = ['green' if v > 0.8 else 'orange' if v > 0.5 else 'red' for v in quality_values]
            
            bars = ax3.bar(quality_metrics, quality_values, color=colors, alpha=0.7)
            ax3.set_ylabel('得分/比率')
            ax3.set_title('数据质量指标')
            ax3.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars, quality_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # 4. 模型信息
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            info_text = f"""
模型信息:
• 模型名称: {validation_result.model_name}
• 验证类型: {validation_result.validation_type}
• 验证时间: {validation_result.timestamp[:19]}

模型结构:
• 结构有效性: {'是' if validation_result.structure_valid else '否'}
• 参数数量: {validation_result.parameter_count}
• 复杂度得分: {validation_result.complexity_score:.2f}

验证状态:
• 整体状态: {'通过' if validation_result.is_valid else '失败'}
• 验证得分: {validation_result.validation_score:.4f}
• 错误数量: {len(validation_result.errors)}
• 警告数量: {len(validation_result.warnings)}
            """
            
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"可视化报告已保存: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"生成可视化报告失败: {str(e)}")
            raise ValidationError(f"生成可视化报告失败: {str(e)}")
    
    def export_results(self, validation_result: ValidationResult, 
                      formats: List[str] = ['json', 'txt']) -> Dict[str, str]:
        """
        导出验证结果
        
        Args:
            validation_result: 验证结果
            formats: 导出格式列表
            
        Returns:
            导出文件路径字典
        """
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            for format_type in formats:
                if format_type.lower() == 'json':
                    filename = f"validation_result_{validation_result.model_name}_{timestamp}.json"
                    filepath = self.output_dir / filename
                    validation_result.to_json(str(filepath))
                    exported_files['json'] = str(filepath)
                
                elif format_type.lower() == 'txt':
                    filename = f"validation_report_{validation_result.model_name}_{timestamp}.txt"
                    filepath = self.output_dir / filename
                    report_content = validation_result.generate_report()
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                    exported_files['txt'] = str(filepath)
                
                elif format_type.lower() == 'png':
                    exported_files['png'] = self.generate_visual_report(validation_result)
            
            logger.info(f"验证结果导出完成: {list(exported_files.keys())}")
            
        except Exception as e:
            logger.error(f"导出验证结果失败: {str(e)}")
            raise ValidationError(f"导出验证结果失败: {str(e)}")
        
        return exported_files


class ModelValidator:
    """
    V2模型验证器主类
    
    这是主要的模型验证器类，集成了所有验证功能。
    提供统一的接口进行全面的模型验证。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化模型验证器
        
        Args:
            config: 全局配置参数
        """
        self.config = config or {}
        
        # 初始化各个验证器
        self.data_validator = DataValidator(self.config.get('data_validation', {}))
        self.structure_validator = ModelStructureValidator(self.config.get('structure_validation', {}))
        self.cross_validator = CrossValidator(self.config.get('cross_validation', {}))
        self.reporter = ValidationReporter(self.config.get('reporting', {}))
        
        # 验证历史
        self.validation_history: List[ValidationResult] = []
        
        logger.info("V2模型验证器初始化完成")
    
    def validate_model(self, model: Any, X: Union[pd.DataFrame, np.ndarray], 
                      y: Optional[Union[pd.Series, np.ndarray]] = None,
                      validation_types: Optional[List[str]] = None,
                      **kwargs) -> ValidationResult:
        """
        执行完整的模型验证
        
        Args:
            model: 待验证的模型
            X: 特征数据
            y: 目标数据（可选）
            validation_types: 验证类型列表
            **kwargs: 其他参数
            
        Returns:
            验证结果
        """
        # 默认验证类型
        if validation_types is None:
            validation_types = ['data_quality', 'structure', 'performance', 'cross_validation']
        
        # 获取模型名称
        model_name = kwargs.get('model_name', type(model).__name__)
        validation_type = '+'.join(validation_types)
        
        # 创建验证结果对象
        result = ValidationResult(
            model_name=model_name,
            validation_type=validation_type
        )
        
        try:
            logger.info(f"开始验证模型: {model_name}")
            start_time = time.time()
            
            # 1. 数据质量验证
            if 'data_quality' in validation_types:
                logger.info("执行数据质量验证...")
                data_quality = self.data_validator.validate_data_quality(X, y)
                
                result.data_quality_score = data_quality['quality_score']
                result.missing_values_ratio = data_quality['missing_values_ratio']
                result.duplicate_ratio = data_quality['duplicate_ratio']
                result.outlier_ratio = data_quality['outlier_ratio']
                
                result.warnings.extend(data_quality.get('warnings', []))
                result.errors.extend(data_quality.get('errors', []))
                result.validation_details['data_quality'] = data_quality
                
                if not data_quality['is_valid']:
                    result.is_valid = False
                    result.errors.append("数据质量验证失败")
            
            # 2. 模型结构验证
            if 'structure' in validation_types:
                logger.info("执行模型结构验证...")
                X_shape = X.shape if hasattr(X, 'shape') else (len(X),)
                structure_validation = self.structure_validator.validate_model_structure(model, X_shape)
                
                result.structure_valid = structure_validation['structure_valid']
                result.parameter_count = structure_validation['parameter_count']
                result.complexity_score = structure_validation['complexity_score']
                
                result.warnings.extend(structure_validation.get('warnings', []))
                result.errors.extend(structure_validation.get('errors', []))
                result.validation_details['structure'] = structure_validation
                
                if not structure_validation['is_valid']:
                    result.is_valid = False
                    result.errors.append("模型结构验证失败")
            
            # 3. 性能验证（如果有目标变量）
            if 'performance' in validation_types and y is not None:
                logger.info("执行性能验证...")
                performance_result = self._validate_performance(model, X, y, **kwargs)
                
                # 更新性能指标
                for attr in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 
                           'mse', 'mae', 'r2_score']:
                    if hasattr(performance_result, attr):
                        setattr(result, attr, getattr(performance_result, attr))
                
                result.warnings.extend(performance_result.get('warnings', []))
                result.errors.extend(performance_result.get('errors', []))
                result.validation_details['performance'] = performance_result
            
            # 4. 交叉验证
            if 'cross_validation' in validation_types and y is not None:
                logger.info("执行交叉验证...")
                cv_method = kwargs.get('cv_method', 'kfold')
                cv_result = self.cross_validator.cross_validate(model, X, y, cv_method)
                
                result.cv_scores = cv_result.get('cv_scores', [])
                result.cv_mean = cv_result.get('cv_mean', 0.0)
                result.cv_std = cv_result.get('cv_std', 0.0)
                
                result.warnings.extend(cv_result.get('warnings', []))
                result.errors.extend(cv_result.get('errors', []))
                result.validation_details['cross_validation'] = cv_result
            
            # 计算综合验证得分
            result.validation_score = self._calculate_validation_score(result)
            
            # 记录验证时间
            validation_time = time.time() - start_time
            result.validation_details['validation_time'] = validation_time
            result.info_messages.append(f"验证完成，耗时: {validation_time:.2f}秒")
            
            # 添加到历史记录
            self.validation_history.append(result)
            
            logger.info(f"模型验证完成: {model_name}, 得分: {result.validation_score:.4f}")
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"验证过程发生错误: {str(e)}")
            logger.error(f"模型验证错误: {str(e)}")
        
        return result
    
    def _validate_performance(self, model: Any, X: Union[pd.DataFrame, np.ndarray], 
                            y: Union[pd.Series, np.ndarray], 
                            test_size: float = 0.2, **kwargs) -> Dict[str, Any]:
        """
        验证模型性能
        
        Args:
            model: 待验证的模型
            X: 特征数据
            y: 目标数据
            test_size: 测试集比例
            **kwargs: 其他参数
            
        Returns:
            性能验证结果
        """
        result = {
            'warnings': [],
            'errors': [],
            'details': {}
        }
        
        try:
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, 
                random_state=kwargs.get('random_state', 42),
                stratify=y if len(np.unique(y)) < len(y) * 0.5 else None
            )
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 判断问题类型
            is_classification = len(np.unique(y)) < len(y) * 0.5
            
            if is_classification:
                # 分类问题指标
                try:
                    result['accuracy'] = accuracy_score(y_test, y_pred)
                except:
                    pass
                
                try:
                    result['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                except:
                    pass
                
                try:
                    result['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                except:
                    pass
                
                try:
                    result['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                except:
                    pass
                
                # ROC-AUC（仅适用于二分类）
                try:
                    if len(np.unique(y)) == 2:
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test)[:, 1]
                            result['roc_auc'] = roc_auc_score(y_test, y_proba)
                except:
                    pass
                
            else:
                # 回归问题指标
                try:
                    result['mse'] = mean_squared_error(y_test, y_pred)
                except:
                    pass
                
                try:
                    result['mae'] = mean_absolute_error(y_test, y_pred)
                except:
                    pass
                
                try:
                    result['r2_score'] = r2_score(y_test, y_pred)
                except:
                    pass
            
            result['details'] = {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'problem_type': 'classification' if is_classification else 'regression',
                'n_classes': len(np.unique(y)) if is_classification else None
            }
            
        except Exception as e:
            result['errors'].append(f"性能验证失败: {str(e)}")
            logger.error(f"性能验证错误: {str(e)}")
        
        return result
    
    def _calculate_validation_score(self, result: ValidationResult) -> float:
        """
        计算综合验证得分
        
        Args:
            result: 验证结果
            
        Returns:
            综合验证得分
        """
        score = 0.0
        weights = []
        
        # 数据质量得分 (权重: 0.2)
        if result.data_quality_score > 0:
            score += result.data_quality_score * 0.2
            weights.append(0.2)
        
        # 模型结构得分 (权重: 0.2)
        if result.structure_valid:
            score += 1.0 * 0.2
            weights.append(0.2)
        
        # 性能得分 (权重: 0.4)
        performance_scores = []
        if result.accuracy is not None:
            performance_scores.append(result.accuracy)
        if result.f1_score is not None:
            performance_scores.append(result.f1_score)
        if result.r2_score is not None:
            performance_scores.append(max(0, result.r2_score))  # R²可能为负
        
        if performance_scores:
            avg_performance = np.mean(performance_scores)
            score += avg_performance * 0.4
            weights.append(0.4)
        
        # 交叉验证得分 (权重: 0.2)
        if result.cv_mean is not None:
            score += result.cv_mean * 0.2
            weights.append(0.2)
        
        # 错误惩罚
        error_penalty = len(result.errors) * 0.1
        score = max(0, score - error_penalty)
        
        # 标准化得分
        if weights:
            score = score / sum(weights)
        
        return min(1.0, max(0.0, score))
    
    def batch_validate(self, models: List[Tuple[str, Any]], 
                      X: Union[pd.DataFrame, np.ndarray],
                      y: Optional[Union[pd.Series, np.ndarray]] = None,
                      **kwargs) -> List[ValidationResult]:
        """
        批量验证多个模型
        
        Args:
            models: 模型列表，每个元素为 (名称, 模型实例) 的元组
            X: 特征数据
            y: 目标数据（可选）
            **kwargs: 其他参数
            
        Returns:
            验证结果列表
        """
        results = []
        
        logger.info(f"开始批量验证 {len(models)} 个模型")
        
        for model_name, model in models:
            try:
                logger.info(f"验证模型: {model_name}")
                result = self.validate_model(
                    model=model,
                    X=X,
                    y=y,
                    model_name=model_name,
                    **kwargs
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"批量验证中模型 {model_name} 失败: {str(e)}")
                # 创建错误结果
                error_result = ValidationResult(
                    model_name=model_name,
                    validation_type="error",
                    is_valid=False
                )
                error_result.errors.append(f"验证失败: {str(e)}")
                results.append(error_result)
        
        logger.info(f"批量验证完成，成功验证 {len([r for r in results if r.is_valid])} 个模型")
        
        return results
    
    def compare_models(self, results: List[ValidationResult]) -> pd.DataFrame:
        """
        比较多个模型的验证结果
        
        Args:
            results: 验证结果列表
            
        Returns:
            比较结果DataFrame
        """
        comparison_data = []
        
        for result in results:
            row = {
                '模型名称': result.model_name,
                '验证状态': '通过' if result.is_valid else '失败',
                '综合得分': result.validation_score,
                '数据质量得分': result.data_quality_score,
                '结构有效性': result.structure_valid,
                '参数数量': result.parameter_count,
                '准确率': result.accuracy,
                'F1得分': result.f1_score,
                'R²得分': result.r2_score,
                '交叉验证均值': result.cv_mean,
                '交叉验证标准差': result.cv_std,
                '错误数量': len(result.errors),
                '警告数量': len(result.warnings)
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # 按综合得分排序
        df = df.sort_values('综合得分', ascending=False)
        
        logger.info("模型比较完成")
        
        return df
    
    def get_validation_history(self, model_name: Optional[str] = None) -> List[ValidationResult]:
        """
        获取验证历史
        
        Args:
            model_name: 模型名称过滤（可选）
            
        Returns:
            验证历史列表
        """
        if model_name:
            return [r for r in self.validation_history if r.model_name == model_name]
        else:
            return self.validation_history.copy()
    
    def clear_history(self):
        """清空验证历史"""
        self.validation_history.clear()
        logger.info("验证历史已清空")


# 使用示例和测试函数
def example_usage():
    """使用示例"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # 生成示例数据
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                              n_informative=15, n_redundant=5, random_state=42)
    
    # 创建模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 创建验证器配置
    config = {
        'data_validation': {
            'quality_threshold': 0.8,
            'missing_threshold': 0.1
        },
        'cross_validation': {
            'cv_folds': 5,
            'scoring': 'accuracy'
        },
        'reporting': {
            'output_dir': './validation_reports'
        }
    }
    
    # 创建验证器
    validator = ModelValidator(config)
    
    # 执行验证
    result = validator.validate_model(
        model=model,
        X=X,
        y=y,
        model_name="RandomForest示例",
        validation_types=['data_quality', 'structure', 'performance', 'cross_validation']
    )
    
    # 生成报告
    print(result.generate_report())
    
    # 导出结果
    exported_files = validator.reporter.export_results(result, ['json', 'txt', 'png'])
    print(f"报告已导出: {list(exported_files.keys())}")
    
    return result


def test_model_validator():
    """测试函数"""
    print("开始测试V2模型验证器...")
    
    try:
        # 测试基本功能
        result = example_usage()
        
        # 测试批量验证
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=500, n_features=10, n_classes=3, n_informative=8, random_state=42)
        
        models = [
            ("RandomForest", RandomForestClassifier(random_state=42)),
            ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
            ("SVM", SVC(random_state=42))
        ]
        
        validator = ModelValidator()
        batch_results = validator.batch_validate(models, X, y)
        
        # 比较模型
        comparison_df = validator.compare_models(batch_results)
        print("\n模型比较结果:")
        print(comparison_df.to_string(index=False))
        
        print("\n✅ 所有测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        raise


if __name__ == "__main__":
    # 运行测试
    test_model_validator()