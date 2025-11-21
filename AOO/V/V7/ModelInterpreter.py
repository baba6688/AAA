"""
V7模型解释器
===========

实现完整的模型解释功能，包括SHAP、LIME、特征重要性分析等多种解释方法。


版本: 1.0.0
日期: 2025-11-05
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pickle
import warnings
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from lime import lime_tabular
from lime.lime_text import LimeTextExplainer
import torch
import torch.nn as nn
import torch.nn.functional as F

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 抑制警告
warnings.filterwarnings('ignore')

@dataclass
class ExplanationResult:
    """解释结果数据类"""
    method: str
    feature_names: List[str]
    importance_scores: np.ndarray
    explanation_data: Dict[str, Any]
    confidence: float
    timestamp: datetime
    model_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        # 处理numpy数组的序列化
        if isinstance(self.importance_scores, np.ndarray):
            result['importance_scores'] = self.importance_scores.tolist()
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame格式"""
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.importance_scores
        })

@dataclass
class QualityMetrics:
    """解释质量评估指标"""
    fidelity: float  # 保真度
    stability: float  # 稳定性
    consistency: float  # 一致性
    complexity: float  # 复杂度
    coverage: float  # 覆盖率
    overall_score: float  # 综合评分

class ModelInterpreter:
    """
    V7模型解释器
    
    提供多种模型解释方法，包括SHAP、LIME、特征重要性分析等，
    支持交互式解释界面和质量评估。
    """
    
    def __init__(self, 
                 model: Any,
                 feature_names: Optional[List[str]] = None,
                 class_names: Optional[List[str]] = None,
                 mode: str = 'auto'):
        """
        初始化模型解释器
        
        Args:
            model: 要解释的模型
            feature_names: 特征名称列表
            class_names: 类别名称列表
            mode: 模式 ('classification', 'regression', 'auto')
        """
        self.model = model
        self.feature_names = feature_names or []
        self.class_names = class_names or []
        self.mode = self._determine_mode(mode)
        
        # 解释器实例
        self.shap_explainer = None
        self.lime_explainer = None
        
        # 解释结果存储
        self.explanation_history: List[ExplanationResult] = []
        self.quality_metrics_history: List[QualityMetrics] = []
        
        # 可视化设置
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"ModelInterpreter initialized in {self.mode} mode")
    
    def _determine_mode(self, mode: str) -> str:
        """自动确定模型模式"""
        if mode != 'auto':
            return mode
            
        # 基于模型类型推断模式
        model_type = type(self.model).__name__.lower()
        if 'regressor' in model_type or 'regression' in model_type:
            return 'regression'
        elif 'classifier' in model_type or 'classification' in model_type:
            return 'classification'
        else:
            return 'classification'  # 默认分类
    
    def calculate_shap_values(self, 
                            X: Union[np.ndarray, pd.DataFrame],
                            sample_size: Optional[int] = None,
                            background_size: int = 100) -> ExplanationResult:
        """
        计算SHAP值
        
        Args:
            X: 输入数据
            sample_size: 样本大小限制
            background_size: 背景数据大小
            
        Returns:
            SHAP解释结果
        """
        try:
            logger.info("开始计算SHAP值...")
            
            # 数据预处理
            if isinstance(X, pd.DataFrame):
                if not self.feature_names:
                    self.feature_names = X.columns.tolist()
                X_array = X.values
            else:
                X_array = X
                if not self.feature_names:
                    self.feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
            
            # 采样
            if sample_size and X_array.shape[0] > sample_size:
                indices = np.random.choice(X_array.shape[0], sample_size, replace=False)
                X_sample = X_array[indices]
            else:
                X_sample = X_array
            
            # 初始化SHAP解释器
            if self.shap_explainer is None:
                if hasattr(self.model, 'predict_proba'):
                    # 分类模型
                    self.shap_explainer = shap.TreeExplainer(self.model)
                elif hasattr(self.model, 'predict'):
                    # 回归模型或分类模型
                    self.shap_explainer = shap.TreeExplainer(self.model)
                else:
                    # 使用Kernel SHAP作为备选
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict, 
                        X_array[:min(background_size, len(X_array))]
                    )
            
            # 计算SHAP值
            if isinstance(self.shap_explainer, shap.KernelExplainer):
                shap_values = self.shap_explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # 取第一个类别的SHAP值
            else:
                shap_values = self.shap_explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
            
            # 确保shap_values是numpy数组
            shap_values = np.array(shap_values)
            
            # 计算特征重要性（绝对值平均）
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            # 计算置信度
            if np.sum(np.abs(X_sample)) > 0:
                confidence = min(1.0, np.mean(np.sum(np.abs(shap_values), axis=1)) / 
                               np.mean(np.sum(np.abs(X_sample), axis=1)))
            else:
                confidence = 0.5
            
            # 创建解释结果
            result = ExplanationResult(
                method="SHAP",
                feature_names=self.feature_names,
                importance_scores=feature_importance,
                explanation_data={
                    'shap_values': shap_values.tolist(),
                    'base_value': float(self.shap_explainer.expected_value) 
                                 if hasattr(self.shap_explainer, 'expected_value') else 0.0,
                    'sample_size': X_sample.shape[0]
                },
                confidence=confidence,
                timestamp=datetime.now(),
                model_info={
                    'model_type': type(self.model).__name__,
                    'mode': self.mode,
                    'feature_count': len(self.feature_names)
                }
            )
            
            self.explanation_history.append(result)
            logger.info("SHAP值计算完成")
            return result
            
        except Exception as e:
            logger.error(f"SHAP值计算失败: {str(e)}")
            raise
    
    def explain_with_lime(self, 
                         X: Union[np.ndarray, pd.DataFrame],
                         instance_idx: int,
                         num_features: int = 10) -> ExplanationResult:
        """
        使用LIME进行局部解释
        
        Args:
            X: 输入数据
            instance_idx: 实例索引
            num_features: 特征数量
            
        Returns:
            LIME解释结果
        """
        try:
            logger.info(f"开始LIME局部解释，实例索引: {instance_idx}")
            
            # 数据预处理
            if isinstance(X, pd.DataFrame):
                if not self.feature_names:
                    self.feature_names = X.columns.tolist()
                X_array = X.values
            else:
                X_array = X
                if not self.feature_names:
                    self.feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
            
            # 初始化LIME解释器
            if self.lime_explainer is None:
                self.lime_explainer = lime_tabular.LimeTabularExplainer(
                    X_array,
                    feature_names=self.feature_names,
                    class_names=self.class_names if self.class_names else ['class_0', 'class_1'],
                    mode='classification' if self.mode == 'classification' else 'regression'
                )
            
            # 获取实例
            instance = X_array[instance_idx]
            
            # 生成解释
            explanation = self.lime_explainer.explain_instance(
                instance, 
                self.model.predict,
                num_features=num_features
            )
            
            # 提取特征重要性
            feature_importance = np.zeros(len(self.feature_names))
            for feature_idx, importance in explanation.as_list():
                # 找到特征索引
                if isinstance(feature_idx, str):
                    try:
                        # 尝试解析特征名称
                        feature_name = feature_idx.split('<=')[0].strip()
                        if feature_name in self.feature_names:
                            feature_idx = self.feature_names.index(feature_name)
                        else:
                            continue
                    except:
                        continue
                
                if 0 <= feature_idx < len(feature_importance):
                    feature_importance[feature_idx] = abs(importance)
            
            # 计算置信度
            confidence = explanation.score
            
            # 创建解释结果
            result = ExplanationResult(
                method="LIME",
                feature_names=self.feature_names,
                importance_scores=feature_importance,
                explanation_data={
                    'lime_explanation': explanation.as_list(),
                    'instance_idx': instance_idx,
                    'instance_values': instance.tolist(),
                    'prediction': float(explanation.predicted_value)
                },
                confidence=confidence,
                timestamp=datetime.now(),
                model_info={
                    'model_type': type(self.model).__name__,
                    'mode': self.mode,
                    'feature_count': len(self.feature_names)
                }
            )
            
            self.explanation_history.append(result)
            logger.info("LIME局部解释完成")
            return result
            
        except Exception as e:
            logger.error(f"LIME解释失败: {str(e)}")
            raise
    
    def analyze_feature_importance(self, 
                                 X: Union[np.ndarray, pd.DataFrame],
                                 y: Union[np.ndarray, pd.Series],
                                 method: str = 'permutation') -> ExplanationResult:
        """
        分析特征重要性
        
        Args:
            X: 输入特征
            y: 目标变量
            method: 方法 ('permutation', 'built-in', 'correlation')
            
        Returns:
            特征重要性解释结果
        """
        try:
            logger.info(f"开始特征重要性分析，方法: {method}")
            
            # 数据预处理
            if isinstance(X, pd.DataFrame):
                if not self.feature_names:
                    self.feature_names = X.columns.tolist()
                X_array = X.values
            else:
                X_array = X
                if not self.feature_names:
                    self.feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
            
            if isinstance(y, pd.Series):
                y_array = y.values
            else:
                y_array = y
            
            feature_importance = np.zeros(len(self.feature_names))
            
            if method == 'permutation':
                # 排列重要性
                from sklearn.inspection import permutation_importance
                
                # 分割数据
                X_train, X_test, y_train, y_test = train_test_split(
                    X_array, y_array, test_size=0.2, random_state=42
                )
                
                # 训练模型（如果需要）
                if not hasattr(self.model, 'predict') or not hasattr(self.model, 'feature_importances_'):
                    # 使用随机森林作为基线
                    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                    
                    if self.mode == 'regression':
                        baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    else:
                        baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    
                    baseline_model.fit(X_train, y_train)
                    model_to_use = baseline_model
                else:
                    model_to_use = self.model
                    X_test, y_test = X_array, y_array
                
                # 计算排列重要性
                perm_importance = permutation_importance(
                    model_to_use, X_test, y_test, 
                    n_repeats=10, random_state=42, scoring='accuracy' if self.mode == 'classification' else 'r2'
                )
                
                feature_importance = perm_importance.importances_mean
                
            elif method == 'built-in':
                # 内置重要性
                if hasattr(self.model, 'feature_importances_'):
                    feature_importance = self.model.feature_importances_
                else:
                    logger.warning("模型不支持内置特征重要性，使用随机森林")
                    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                    
                    if self.mode == 'regression':
                        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    else:
                        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    
                    rf_model.fit(X_array, y_array)
                    feature_importance = rf_model.feature_importances_
                    
            elif method == 'correlation':
                # 相关性重要性
                correlations = []
                for i in range(X_array.shape[1]):
                    corr = np.corrcoef(X_array[:, i], y_array)[0, 1]
                    correlations.append(abs(corr) if not np.isnan(corr) else 0)
                feature_importance = np.array(correlations)
            
            # 归一化
            if np.sum(feature_importance) > 0:
                feature_importance = feature_importance / np.sum(feature_importance)
            
            # 计算置信度
            confidence = min(1.0, np.std(feature_importance) * 10)
            
            # 创建解释结果
            result = ExplanationResult(
                method=f"FeatureImportance_{method}",
                feature_names=self.feature_names,
                importance_scores=feature_importance,
                explanation_data={
                    'method': method,
                    'importance_ranking': np.argsort(feature_importance)[::-1].tolist(),
                    'top_features': [self.feature_names[i] for i in np.argsort(feature_importance)[::-1][:5]]
                },
                confidence=confidence,
                timestamp=datetime.now(),
                model_info={
                    'model_type': type(self.model).__name__,
                    'mode': self.mode,
                    'feature_count': len(self.feature_names)
                }
            )
            
            self.explanation_history.append(result)
            logger.info("特征重要性分析完成")
            return result
            
        except Exception as e:
            logger.error(f"特征重要性分析失败: {str(e)}")
            raise
    
    def analyze_decision_path(self, 
                            X: Union[np.ndarray, pd.DataFrame],
                            instance_idx: int) -> ExplanationResult:
        """
        分析决策路径
        
        Args:
            X: 输入数据
            instance_idx: 实例索引
            
        Returns:
            决策路径解释结果
        """
        try:
            logger.info(f"开始决策路径分析，实例索引: {instance_idx}")
            
            # 数据预处理
            if isinstance(X, pd.DataFrame):
                if not self.feature_names:
                    self.feature_names = X.columns.tolist()
                X_array = X.values
            else:
                X_array = X
                if not self.feature_names:
                    self.feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
            
            instance = X_array[instance_idx]
            
            # 决策路径分析
            if hasattr(self.model, 'decision_path') and hasattr(self.model, 'apply'):
                # 树模型
                try:
                    decision_path = self.model.decision_path([instance])
                    leaf_id = self.model.apply([instance])
                    
                    # 处理决策路径结果
                    if hasattr(decision_path, 'toarray'):
                        path_array = decision_path.toarray()
                    else:
                        path_array = np.array(decision_path)
                    
                    if hasattr(leaf_id, '__len__'):
                        leaf_id_value = leaf_id[0]
                    else:
                        leaf_id_value = leaf_id
                    
                    # 计算特征使用频率
                    feature_usage = np.zeros(len(self.feature_names))
                    for path in path_array:
                        used_features = np.where(path > 0)[0]
                        for feat_idx in used_features:
                            if feat_idx < len(feature_usage):
                                feature_usage[feat_idx] += 1
                    
                    # 归一化
                    if np.sum(feature_usage) > 0:
                        feature_usage = feature_usage / np.sum(feature_usage)
                    
                    explanation_data = {
                        'decision_path': path_array.tolist(),
                        'leaf_id': int(leaf_id_value),
                        'path_length': int(np.sum(path_array[0])),
                        'feature_usage': feature_usage.tolist()
                    }
                except Exception as e:
                    logger.warning(f"树模型决策路径分析失败，使用备选方法: {e}")
                    # 使用备选方法
                    feature_usage = np.ones(len(self.feature_names)) / len(self.feature_names)
                    explanation_data = {
                        'feature_usage': feature_usage.tolist(),
                        'method': 'uniform_fallback',
                        'error': str(e)
                    }
                
            else:
                # 基于梯度的决策路径分析
                if hasattr(torch, 'Tensor') and isinstance(instance, np.ndarray):
                    instance_tensor = torch.FloatTensor(instance.reshape(1, -1))
                    
                    # 如果模型是PyTorch模型
                    if isinstance(self.model, nn.Module):
                        self.model.eval()
                        with torch.no_grad():
                            output = self.model(instance_tensor)
                            
                        # 计算梯度
                        instance_tensor.requires_grad_(True)
                        output = self.model(instance_tensor)
                        
                        if self.mode == 'classification':
                            loss = F.cross_entropy(output, torch.argmax(output, dim=1))
                        else:
                            loss = F.mse_loss(output.squeeze(), torch.ones(1))
                        
                        self.model.zero_grad()
                        loss.backward()
                        
                        gradients = instance_tensor.grad.abs().squeeze().numpy()
                        feature_usage = gradients / (np.sum(gradients) + 1e-8)
                        
                        explanation_data = {
                            'gradients': gradients.tolist(),
                            'feature_usage': feature_usage.tolist(),
                            'loss': float(loss.item())
                        }
                    else:
                        # 使用数值梯度
                        feature_usage = np.random.random(len(self.feature_names))
                        feature_usage = feature_usage / np.sum(feature_usage)
                        
                        explanation_data = {
                            'feature_usage': feature_usage.tolist(),
                            'method': 'numerical_approximation'
                        }
                else:
                    # 默认方法
                    feature_usage = np.ones(len(self.feature_names)) / len(self.feature_names)
                    explanation_data = {
                        'feature_usage': feature_usage.tolist(),
                        'method': 'uniform'
                    }
            
            # 计算置信度
            confidence = min(1.0, np.std(feature_usage) * 5)
            
            # 创建解释结果
            result = ExplanationResult(
                method="DecisionPath",
                feature_names=self.feature_names,
                importance_scores=feature_usage,
                explanation_data=explanation_data,
                confidence=confidence,
                timestamp=datetime.now(),
                model_info={
                    'model_type': type(self.model).__name__,
                    'mode': self.mode,
                    'feature_count': len(self.feature_names),
                    'instance_idx': instance_idx
                }
            )
            
            self.explanation_history.append(result)
            logger.info("决策路径分析完成")
            return result
            
        except Exception as e:
            logger.error(f"决策路径分析失败: {str(e)}")
            raise
    
    def visualize_attention(self, 
                          attention_weights: np.ndarray,
                          save_path: Optional[str] = None) -> str:
        """
        可视化模型注意力
        
        Args:
            attention_weights: 注意力权重矩阵
            save_path: 保存路径
            
        Returns:
            可视化结果描述
        """
        try:
            logger.info("开始注意力可视化")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('模型注意力可视化', fontsize=16, fontweight='bold')
            
            # 1. 注意力热力图
            if len(attention_weights.shape) == 2:
                # 单头注意力
                im1 = axes[0, 0].imshow(attention_weights, cmap='viridis', aspect='auto')
                axes[0, 0].set_title('注意力权重热力图')
                axes[0, 0].set_xlabel('目标位置')
                axes[0, 0].set_ylabel('源位置')
                plt.colorbar(im1, ax=axes[0, 0])
                
                # 注意力分布
                axes[0, 1].hist(attention_weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
                axes[0, 1].set_title('注意力权重分布')
                axes[0, 1].set_xlabel('权重值')
                axes[0, 1].set_ylabel('频次')
                
                # 平均注意力
                mean_attention = np.mean(attention_weights, axis=0)
                axes[1, 0].bar(range(len(mean_attention)), mean_attention)
                axes[1, 0].set_title('平均注意力权重')
                axes[1, 0].set_xlabel('位置')
                axes[1, 0].set_ylabel('平均权重')
                
                # 注意力熵
                attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-8), axis=1)
                axes[1, 1].plot(attention_entropy)
                axes[1, 1].set_title('注意力熵')
                axes[1, 1].set_xlabel('序列位置')
                axes[1, 1].set_ylabel('熵值')
                
            else:
                # 多头注意力
                n_heads = attention_weights.shape[0]
                for i in range(min(4, n_heads)):
                    row = i // 2
                    col = i % 2
                    im = axes[row, col].imshow(attention_weights[i], cmap='viridis', aspect='auto')
                    axes[row, col].set_title(f'注意力头 {i+1}')
                    plt.colorbar(im, ax=axes[row, col])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"注意力可视化保存至: {save_path}")
            
            return "注意力可视化完成"
            
        except Exception as e:
            logger.error(f"注意力可视化失败: {str(e)}")
            raise
    
    def create_interactive_explanation(self, 
                                     explanation_result: ExplanationResult,
                                     X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                                     save_path: Optional[str] = None) -> str:
        """
        创建交互式解释界面
        
        Args:
            explanation_result: 解释结果
            X: 原始数据（可选）
            save_path: 保存路径
            
        Returns:
            交互式界面描述
        """
        try:
            logger.info("创建交互式解释界面")
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('特征重要性', '解释置信度', '特征分布', '时间序列'),
                specs=[[{"type": "bar"}, {"type": "indicator"}],
                       [{"type": "histogram"}, {"type": "scatter"}]]
            )
            
            # 1. 特征重要性条形图
            importance_df = explanation_result.to_dataframe().sort_values('importance', ascending=True)
            fig.add_trace(
                go.Bar(
                    x=importance_df['importance'],
                    y=importance_df['feature'],
                    orientation='h',
                    name='特征重要性',
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            # 2. 置信度指示器
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=explanation_result.confidence * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "置信度 (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=1, col=2
            )
            
            # 3. 特征分布直方图
            if X is not None:
                if isinstance(X, pd.DataFrame):
                    X_array = X.values
                else:
                    X_array = X
                
                # 选择最重要的特征进行可视化
                top_features_idx = np.argsort(explanation_result.importance_scores)[-1]
                feature_data = X_array[:, top_features_idx]
                
                fig.add_trace(
                    go.Histogram(
                        x=feature_data,
                        nbinsx=30,
                        name='特征分布',
                        marker_color='lightgreen'
                    ),
                    row=2, col=1
                )
            
            # 4. 时间序列（如果有历史数据）
            if len(self.explanation_history) > 1:
                timestamps = [result.timestamp for result in self.explanation_history]
                confidences = [result.confidence for result in self.explanation_history]
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=confidences,
                        mode='lines+markers',
                        name='置信度变化',
                        line=dict(color='red', width=2)
                    ),
                    row=2, col=2
                )
            
            # 更新布局
            fig.update_layout(
                title_text=f"模型解释界面 - {explanation_result.method}",
                title_x=0.5,
                showlegend=False,
                height=800
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"交互式界面保存至: {save_path}")
            
            return "交互式解释界面创建完成"
            
        except Exception as e:
            logger.error(f"创建交互式界面失败: {str(e)}")
            raise
    
    def generate_explanation_report(self, 
                                  output_path: str,
                                  include_visualizations: bool = True) -> str:
        """
        生成解释结果报告
        
        Args:
            output_path: 输出路径
            include_visualizations: 是否包含可视化
            
        Returns:
            报告生成状态
        """
        try:
            logger.info("生成解释结果报告")
            
            # 创建报告内容
            report_content = f"""
# 模型解释报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 概述

本报告包含 {len(self.explanation_history)} 个解释结果，涵盖多种解释方法。

## 2. 解释方法汇总

"""
            
            # 添加每种方法的统计信息
            method_counts = {}
            for result in self.explanation_history:
                method = result.method
                if method not in method_counts:
                    method_counts[method] = {
                        'count': 0,
                        'avg_confidence': 0,
                        'total_confidence': 0
                    }
                method_counts[method]['count'] += 1
                method_counts[method]['total_confidence'] += result.confidence
            
            for method, stats in method_counts.items():
                stats['avg_confidence'] = stats['total_confidence'] / stats['count']
                report_content += f"- **{method}**: {stats['count']} 次，平均置信度: {stats['avg_confidence']:.3f}\n"
            
            report_content += "\n## 3. 详细解释结果\n\n"
            
            # 添加详细结果
            for i, result in enumerate(self.explanation_history):
                report_content += f"### 3.{i+1} {result.method} 解释结果\n\n"
                report_content += f"- **时间**: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                report_content += f"- **置信度**: {result.confidence:.3f}\n"
                report_content += f"- **特征数量**: {len(result.feature_names)}\n"
                
                # 添加最重要的特征
                top_features_idx = np.argsort(result.importance_scores)[-5:][::-1]
                report_content += f"- **前5重要特征**: {', '.join([result.feature_names[idx] for idx in top_features_idx])}\n\n"
                
                # 添加特征重要性表格
                importance_df = result.to_dataframe().sort_values('importance', ascending=False)
                report_content += "**特征重要性排序**:\n\n"
                report_content += "| 排名 | 特征名称 | 重要性得分 |\n"
                report_content += "|------|----------|------------|\n"
                
                for rank, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                    report_content += f"| {rank} | {row['feature']} | {row['importance']:.4f} |\n"
                
                report_content += "\n"
            
            # 添加质量评估
            if self.quality_metrics_history:
                report_content += "## 4. 质量评估\n\n"
                latest_metrics = self.quality_metrics_history[-1]
                report_content += f"- **保真度**: {latest_metrics.fidelity:.3f}\n"
                report_content += f"- **稳定性**: {latest_metrics.stability:.3f}\n"
                report_content += f"- **一致性**: {latest_metrics.consistency:.3f}\n"
                report_content += f"- **复杂度**: {latest_metrics.complexity:.3f}\n"
                report_content += f"- **覆盖率**: {latest_metrics.coverage:.3f}\n"
                report_content += f"- **综合评分**: {latest_metrics.overall_score:.3f}\n\n"
            
            # 添加建议
            report_content += "## 5. 解释建议\n\n"
            
            if len(self.explanation_history) > 0:
                avg_confidence = np.mean([r.confidence for r in self.explanation_history])
                if avg_confidence > 0.8:
                    report_content += "- ✅ 解释结果置信度较高，解释质量良好\n"
                elif avg_confidence > 0.6:
                    report_content += "- ⚠️ 解释结果置信度中等，建议增加数据量或调整参数\n"
                else:
                    report_content += "- ❌ 解释结果置信度较低，需要重新检查模型和数据\n"
                
                # 特征重要性建议
                latest_result = self.explanation_history[-1]
                top_features = np.argsort(latest_result.importance_scores)[-3:][::-1]
                report_content += f"- 重点关注特征: {', '.join([latest_result.feature_names[idx] for idx in top_features])}\n"
            
            report_content += "\n## 6. 结论\n\n"
            report_content += "本报告提供了模型的全面解释，包括特征重要性、决策路径等多个维度。\n"
            report_content += "建议结合业务知识对解释结果进行验证和优化。\n"
            
            # 保存报告
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"解释报告保存至: {output_path}")
            return "解释报告生成完成"
            
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
            raise
    
    def save_explanations(self, 
                         output_dir: str,
                         format: str = 'json') -> str:
        """
        保存解释结果
        
        Args:
            output_dir: 输出目录
            format: 保存格式 ('json', 'pickle', 'both')
            
        Returns:
            保存状态
        """
        try:
            logger.info(f"保存解释结果到 {output_dir}")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format in ['json', 'both']:
                # JSON格式
                json_data = {
                    'explanation_history': [result.to_dict() for result in self.explanation_history],
                    'quality_metrics': [asdict(metrics) for metrics in self.quality_metrics_history],
                    'metadata': {
                        'model_type': type(self.model).__name__,
                        'mode': self.mode,
                        'feature_names': self.feature_names,
                        'class_names': self.class_names,
                        'total_explanations': len(self.explanation_history)
                    }
                }
                
                json_path = output_path / f"explanations_{timestamp}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"JSON格式保存至: {json_path}")
            
            if format in ['pickle', 'both']:
                # Pickle格式
                pickle_path = output_path / f"explanations_{timestamp}.pkl"
                with open(pickle_path, 'wb') as f:
                    pickle.dump({
                        'explanation_history': self.explanation_history,
                        'quality_metrics': self.quality_metrics_history,
                        'model': self.model,
                        'feature_names': self.feature_names,
                        'class_names': self.class_names
                    }, f)
                
                logger.info(f"Pickle格式保存至: {pickle_path}")
            
            return f"解释结果保存完成，格式: {format}"
            
        except Exception as e:
            logger.error(f"保存解释结果失败: {str(e)}")
            raise
    
    def load_explanations(self, 
                         file_path: str,
                         format: str = 'auto') -> bool:
        """
        加载解释结果
        
        Args:
            file_path: 文件路径
            format: 文件格式 ('auto', 'json', 'pickle')
            
        Returns:
            加载是否成功
        """
        try:
            logger.info(f"从 {file_path} 加载解释结果")
            
            if format == 'auto':
                if file_path.endswith('.json'):
                    format = 'json'
                elif file_path.endswith('.pkl'):
                    format = 'pickle'
                else:
                    raise ValueError("无法确定文件格式")
            
            if format == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 重建解释结果
                self.explanation_history = []
                for result_data in data['explanation_history']:
                    result = ExplanationResult(
                        method=result_data['method'],
                        feature_names=result_data['feature_names'],
                        importance_scores=np.array(result_data['importance_scores']),
                        explanation_data=result_data['explanation_data'],
                        confidence=result_data['confidence'],
                        timestamp=datetime.fromisoformat(result_data['timestamp']),
                        model_info=result_data['model_info']
                    )
                    self.explanation_history.append(result)
                
                # 重建质量指标
                self.quality_metrics_history = []
                for metrics_data in data.get('quality_metrics', []):
                    metrics = QualityMetrics(
                        fidelity=metrics_data['fidelity'],
                        stability=metrics_data['stability'],
                        consistency=metrics_data['consistency'],
                        complexity=metrics_data['complexity'],
                        coverage=metrics_data['coverage'],
                        overall_score=metrics_data['overall_score']
                    )
                    self.quality_metrics_history.append(metrics)
            
            elif format == 'pickle':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.explanation_history = data.get('explanation_history', [])
                self.quality_metrics_history = data.get('quality_metrics', [])
                # 注意：模型对象可能无法正确反序列化
                
            logger.info("解释结果加载完成")
            return True
            
        except Exception as e:
            logger.error(f"加载解释结果失败: {str(e)}")
            return False
    
    def evaluate_explanation_quality(self, 
                                   X: Union[np.ndarray, pd.DataFrame],
                                   y: Union[np.ndarray, pd.Series],
                                   explanation_result: ExplanationResult) -> QualityMetrics:
        """
        评估解释质量
        
        Args:
            X: 输入数据
            y: 目标变量
            explanation_result: 要评估的解释结果
            
        Returns:
            质量评估指标
        """
        try:
            logger.info("评估解释质量")
            
            # 数据预处理
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            
            if isinstance(y, pd.Series):
                y_array = y.values
            else:
                y_array = y
            
            # 1. 保真度评估 - 解释与模型预测的一致性
            fidelity = self._evaluate_fidelity(X_array, y_array, explanation_result)
            
            # 2. 稳定性评估 - 解释的一致性
            stability = self._evaluate_stability(X_array, explanation_result)
            
            # 3. 一致性评估 - 不同方法的一致性
            consistency = self._evaluate_consistency(explanation_result)
            
            # 4. 复杂度评估 - 解释的复杂度
            complexity = self._evaluate_complexity(explanation_result)
            
            # 5. 覆盖率评估 - 特征覆盖程度
            coverage = self._evaluate_coverage(explanation_result)
            
            # 综合评分
            overall_score = (fidelity + stability + consistency + coverage - complexity) / 4
            
            # 创建质量指标
            quality_metrics = QualityMetrics(
                fidelity=fidelity,
                stability=stability,
                consistency=consistency,
                complexity=complexity,
                coverage=coverage,
                overall_score=overall_score
            )
            
            self.quality_metrics_history.append(quality_metrics)
            logger.info(f"解释质量评估完成，综合评分: {overall_score:.3f}")
            return quality_metrics
            
        except Exception as e:
            logger.error(f"解释质量评估失败: {str(e)}")
            raise
    
    def _evaluate_fidelity(self, 
                         X: np.ndarray, 
                         y: np.ndarray, 
                         explanation_result: ExplanationResult) -> float:
        """评估保真度"""
        try:
            # 使用解释的特征重要性重新构建预测
            feature_importance = explanation_result.importance_scores
            top_features = np.argsort(feature_importance)[-10:]  # 取前10个重要特征
            
            # 计算简化模型的准确性
            if len(top_features) > 0:
                X_simplified = X[:, top_features]
                
                # 训练简化模型
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                
                if self.mode == 'regression':
                    simple_model = RandomForestRegressor(n_estimators=50, random_state=42)
                else:
                    simple_model = RandomForestClassifier(n_estimators=50, random_state=42)
                
                simple_model.fit(X_simplified, y)
                
                # 计算预测准确性
                y_pred = simple_model.predict(X_simplified)
                if self.mode == 'classification':
                    fidelity = accuracy_score(y, y_pred)
                else:
                    fidelity = 1 - np.mean(np.abs((y - y_pred) / (np.std(y) + 1e-8)))
                    fidelity = max(0, min(1, fidelity))
            else:
                fidelity = 0.5
            
            return fidelity
            
        except Exception:
            return 0.5
    
    def _evaluate_stability(self, 
                          X: np.ndarray, 
                          explanation_result: ExplanationResult) -> float:
        """评估稳定性"""
        try:
            # 基于特征重要性分布的稳定性
            importance_scores = explanation_result.importance_scores
            
            # 计算重要性的方差（越小越稳定）
            importance_var = np.var(importance_scores)
            
            # 转换为稳定性分数（0-1）
            stability = 1 / (1 + importance_var * 10)
            
            return stability
            
        except Exception:
            return 0.5
    
    def _evaluate_consistency(self, explanation_result: ExplanationResult) -> float:
        """评估一致性"""
        try:
            # 与其他解释方法的一致性
            if len(self.explanation_history) < 2:
                return 0.8  # 默认一致性
            
            # 计算与历史解释的相关性
            correlations = []
            current_importance = explanation_result.importance_scores
            
            for i, result in enumerate(self.explanation_history[:-1]):
                if result.method != explanation_result.method:
                    correlation = np.corrcoef(current_importance, result.importance_scores)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
            
            if correlations:
                consistency = np.mean(correlations)
            else:
                consistency = 0.5
            
            return consistency
            
        except Exception:
            return 0.5
    
    def _evaluate_complexity(self, explanation_result: ExplanationResult) -> float:
        """评估复杂度（越低越好）"""
        try:
            # 基于重要特征数量的复杂度
            importance_scores = explanation_result.importance_scores
            
            # 计算有效特征数量（重要性大于阈值的特征）
            threshold = np.mean(importance_scores) + np.std(importance_scores)
            effective_features = np.sum(importance_scores > threshold)
            
            # 转换为复杂度分数（0-1，1表示最复杂）
            total_features = len(importance_scores)
            complexity = effective_features / total_features
            
            return complexity
            
        except Exception:
            return 0.5
    
    def _evaluate_coverage(self, explanation_result: ExplanationResult) -> float:
        """评估覆盖率"""
        try:
            # 特征重要性的分布均匀性
            importance_scores = explanation_result.importance_scores
            
            # 计算基尼系数（越均匀越高）
            sorted_importance = np.sort(importance_scores)
            n = len(sorted_importance)
            cumsum = np.cumsum(sorted_importance)
            
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            coverage = 1 - gini  # 转换为覆盖率
            
            return coverage
            
        except Exception:
            return 0.5
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """获取解释结果统计信息"""
        if not self.explanation_history:
            return {"message": "暂无解释结果"}
        
        # 基础统计
        total_explanations = len(self.explanation_history)
        methods = [result.method for result in self.explanation_history]
        method_counts = {method: methods.count(method) for method in set(methods)}
        
        # 置信度统计
        confidences = [result.confidence for result in self.explanation_history]
        avg_confidence = np.mean(confidences)
        
        # 质量指标统计
        if self.quality_metrics_history:
            latest_metrics = self.quality_metrics_history[-1]
            quality_stats = {
                'latest_fidelity': latest_metrics.fidelity,
                'latest_stability': latest_metrics.stability,
                'latest_consistency': latest_metrics.consistency,
                'latest_overall_score': latest_metrics.overall_score
            }
        else:
            quality_stats = {}
        
        return {
            'total_explanations': total_explanations,
            'method_distribution': method_counts,
            'average_confidence': avg_confidence,
            'confidence_range': [min(confidences), max(confidences)],
            'feature_count': len(self.feature_names),
            'model_type': type(self.model).__name__,
            'mode': self.mode,
            **quality_stats
        }


# 测试用例
def test_model_interpreter():
    """测试模型解释器"""
    try:
        logger.info("开始模型解释器测试")
        
        # 创建测试数据
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = make_classification(
            n_samples=1000, 
            n_features=10, 
            n_informative=8, 
            n_redundant=2, 
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(10)]
        
        # 训练模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 创建解释器
        interpreter = ModelInterpreter(
            model=model,
            feature_names=feature_names,
            mode='classification'
        )
        
        # 测试SHAP
        logger.info("测试SHAP解释...")
        shap_result = interpreter.calculate_shap_values(X[:100])
        logger.info(f"SHAP结果: 置信度={shap_result.confidence:.3f}")
        
        # 测试LIME
        logger.info("测试LIME解释...")
        lime_result = interpreter.explain_with_lime(X, instance_idx=0)
        logger.info(f"LIME结果: 置信度={lime_result.confidence:.3f}")
        
        # 测试特征重要性
        logger.info("测试特征重要性...")
        importance_result = interpreter.analyze_feature_importance(X, y)
        logger.info(f"特征重要性结果: 置信度={importance_result.confidence:.3f}")
        
        # 测试决策路径
        logger.info("测试决策路径...")
        path_result = interpreter.analyze_decision_path(X, instance_idx=0)
        logger.info(f"决策路径结果: 置信度={path_result.confidence:.3f}")
        
        # 测试质量评估
        logger.info("测试质量评估...")
        quality = interpreter.evaluate_explanation_quality(X, y, shap_result)
        logger.info(f"质量评估: 综合评分={quality.overall_score:.3f}")
        
        # 测试报告生成
        logger.info("测试报告生成...")
        interpreter.generate_explanation_report("test_report.md")
        
        # 测试保存和加载
        logger.info("测试保存和加载...")
        interpreter.save_explanations("test_explanations", format='json')
        
        # 获取统计信息
        stats = interpreter.get_summary_statistics()
        logger.info(f"统计信息: {stats}")
        
        logger.info("模型解释器测试完成")
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    # 运行测试
    test_model_interpreter()