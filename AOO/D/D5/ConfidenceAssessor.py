"""
D5置信度评估器
实现多维度置信度评估、置信度校准、不确定性量化等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.stats import chi2, norm, t
import warnings
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from collections import defaultdict, deque

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """置信度等级"""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MEDIUM = "medium"          # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0

class AlertType(Enum):
    """预警类型"""
    LOW_CONFIDENCE = "low_confidence"
    HIGH_UNCERTAINTY = "high_uncertainty"
    MODEL_DEGRADATION = "model_degradation"
    CALIBRATION_DRIFT = "calibration_drift"
    DATA_QUALITY = "data_quality"

@dataclass
class ConfidenceMetrics:
    """置信度指标"""
    overall_confidence: float
    prediction_confidence: float
    model_confidence: float
    data_confidence: float
    temporal_confidence: float
    uncertainty: float
    confidence_interval: Tuple[float, float]
    confidence_level: ConfidenceLevel
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertInfo:
    """预警信息"""
    alert_type: AlertType
    severity: str  # low, medium, high, critical
    message: str
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)

class ConfidenceModel:
    """置信度模型基类"""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def _create_model(self):
        """创建模型"""
        if self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif self.model_type == "logistic":
            return LogisticRegression(random_state=42)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ConfidenceModel':
        """训练模型"""
        self.model = self._create_model()
        self.model.fit(X, y)
        self.is_fitted = True
        
        # 计算特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(
                enumerate(self.model.feature_importances_)
            )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict_proba(X)

class MultiDimensionalConfidenceCalculator:
    """多维度置信度计算器"""
    
    def __init__(self):
        self.weights = {
            'prediction': 0.3,
            'model': 0.25,
            'data': 0.25,
            'temporal': 0.2
        }
        self.calibration_cache = {}
        
    def calculate_prediction_confidence(self, 
                                      predictions: np.ndarray, 
                                      actuals: Optional[np.ndarray] = None,
                                      historical_errors: Optional[np.ndarray] = None) -> float:
        """计算预测置信度"""
        if actuals is not None:
            # 基于实际误差计算
            errors = np.abs(predictions - actuals)
            max_error = np.max(errors) if len(errors) > 0 else 1.0
            confidence = 1.0 - np.mean(errors) / (max_error + 1e-8)
        elif historical_errors is not None:
            # 基于历史误差分布
            error_std = np.std(historical_errors)
            current_error = np.std(predictions)
            confidence = max(0.0, 1.0 - current_error / (error_std + 1e-8))
        else:
            # 基于预测值分布
            prediction_std = np.std(predictions)
            confidence = max(0.0, 1.0 - prediction_std)
        
        return np.clip(confidence, 0.0, 1.0)
    
    def calculate_model_confidence(self, 
                                 model: ConfidenceModel,
                                 X_val: np.ndarray,
                                 y_val: np.ndarray) -> float:
        """计算模型置信度"""
        if not model.is_fitted:
            return 0.0
        
        # 交叉验证评分
        cv_scores = cross_val_score(model.model, X_val, y_val, cv=5)
        cv_confidence = np.mean(cv_scores)
        
        # 特征重要性分散度
        if model.feature_importance:
            importance_values = list(model.feature_importance.values())
            importance_std = np.std(importance_values)
            importance_confidence = 1.0 - importance_std
        else:
            importance_confidence = 0.5
        
        # 综合模型置信度
        model_confidence = 0.7 * cv_confidence + 0.3 * importance_confidence
        return np.clip(model_confidence, 0.0, 1.0)
    
    def calculate_data_confidence(self, 
                                X: np.ndarray,
                                data_quality_metrics: Dict[str, float]) -> float:
        """计算数据置信度"""
        # 基础数据质量指标
        completeness = data_quality_metrics.get('completeness', 1.0)
        consistency = data_quality_metrics.get('consistency', 1.0)
        accuracy = data_quality_metrics.get('accuracy', 1.0)
        
        # 数据分布稳定性
        if X.shape[0] > 1:
            feature_stds = np.std(X, axis=0)
            stability = 1.0 / (1.0 + np.mean(feature_stds))
        else:
            stability = 0.5
        
        # 数据量充足性
        data_sufficiency = min(1.0, X.shape[0] / 100)  # 假设最少需要100个样本
        
        # 综合数据置信度
        data_confidence = (0.3 * completeness + 0.3 * consistency + 
                          0.2 * accuracy + 0.1 * stability + 0.1 * data_sufficiency)
        
        return np.clip(data_confidence, 0.0, 1.0)
    
    def calculate_temporal_confidence(self, 
                                    timestamps: List[datetime],
                                    prediction_horizon: int = 1) -> float:
        """计算时间维度置信度"""
        if len(timestamps) < 2:
            return 0.5
        
        # 时间序列连续性
        time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                     for i in range(1, len(timestamps))]
        
        if not time_diffs:
            return 0.5
        
        # 时间间隔稳定性
        time_std = np.std(time_diffs)
        time_mean = np.mean(time_diffs)
        temporal_stability = 1.0 - (time_std / (time_mean + 1e-8))
        
        # 预测时间合理性
        time_horizon_confidence = max(0.0, 1.0 - prediction_horizon / 365)  # 假设最大预测365天
        
        # 综合时间置信度
        temporal_confidence = 0.6 * temporal_stability + 0.4 * time_horizon_confidence
        
        return np.clip(temporal_confidence, 0.0, 1.0)
    
    def calculate_overall_confidence(self, 
                                   confidence_components: Dict[str, float]) -> float:
        """计算综合置信度"""
        weighted_sum = sum(
            self.weights[component] * confidence 
            for component, confidence in confidence_components.items()
            if component in self.weights
        )
        
        return np.clip(weighted_sum, 0.0, 1.0)

class UncertaintyQuantifier:
    """不确定性量化器"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        
    def calculate_prediction_interval(self, 
                                    predictions: np.ndarray,
                                    residuals: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """计算预测区间"""
        if residuals is None:
            # 使用预测值本身估算不确定性
            pred_std = np.std(predictions)
            mean_pred = np.mean(predictions)
            
            # t分布临界值
            t_critical = stats.t.ppf(1 - self.alpha/2, len(predictions) - 1)
            margin_error = t_critical * pred_std / np.sqrt(len(predictions))
            
            lower_bound = mean_pred - margin_error
            upper_bound = mean_pred + margin_error
        else:
            # 使用残差估算不确定性
            residual_std = np.std(residuals)
            mean_pred = np.mean(predictions)
            
            # t分布临界值
            t_critical = stats.t.ppf(1 - self.alpha/2, len(residuals) - 1)
            margin_error = t_critical * residual_std
            
            lower_bound = mean_pred - margin_error
            upper_bound = mean_pred + margin_error
        
        return (lower_bound, upper_bound)
    
    def calculate_confidence_interval(self, 
                                    estimate: float,
                                    std_error: float,
                                    sample_size: int) -> Tuple[float, float]:
        """计算置信区间"""
        t_critical = stats.t.ppf(1 - self.alpha/2, sample_size - 1)
        margin_error = t_critical * std_error
        
        lower_bound = estimate - margin_error
        upper_bound = estimate + margin_error
        
        return (lower_bound, upper_bound)
    
    def calculate_bootstrap_uncertainty(self, 
                                      predictions: np.ndarray,
                                      n_bootstrap: int = 1000) -> Dict[str, float]:
        """Bootstrap不确定性估算"""
        n_samples = len(predictions)
        bootstrap_predictions = []
        
        # Bootstrap重采样
        for _ in range(n_bootstrap):
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_pred = predictions[bootstrap_indices]
            bootstrap_predictions.append(np.mean(bootstrap_pred))
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # 计算统计量
        uncertainty_metrics = {
            'std': np.std(bootstrap_predictions),
            'mean': np.mean(bootstrap_predictions),
            'q025': np.percentile(bootstrap_predictions, 2.5),
            'q975': np.percentile(bootstrap_predictions, 97.5),
            'iqr': np.percentile(bootstrap_predictions, 75) - np.percentile(bootstrap_predictions, 25)
        }
        
        return uncertainty_metrics
    
    def calculate_ensemble_uncertainty(self, 
                                     ensemble_predictions: List[np.ndarray]) -> Dict[str, float]:
        """集成模型不确定性"""
        if not ensemble_predictions:
            return {}
        
        # 转换为数组
        pred_array = np.array(ensemble_predictions)
        
        # 计算不同类型的不确定性
        aleatoric_uncertainty = np.mean(np.var(pred_array, axis=0))  # 数据不确定性
        epistemic_uncertainty = np.var(np.mean(pred_array, axis=0))  # 模型不确定性
        
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        # 预测一致性
        prediction_spread = np.mean(np.std(pred_array, axis=0))
        
        uncertainty_metrics = {
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'total_uncertainty': total_uncertainty,
            'prediction_spread': prediction_spread,
            'consistency_score': 1.0 / (1.0 + prediction_spread)
        }
        
        return uncertainty_metrics

class ConfidenceCalibrator:
    """置信度校准器"""
    
    def __init__(self, method: str = "platt_scaling"):
        self.method = method
        self.calibration_model = None
        self.is_fitted = False
        self.calibration_metrics = {}
        
    def fit(self, 
          predicted_confidences: np.ndarray, 
          actual_outcomes: np.ndarray) -> 'ConfidenceCalibrator':
        """拟合校准模型"""
        # 确保输入格式正确
        predicted_confidences = np.array(predicted_confidences).reshape(-1, 1)
        actual_outcomes = np.array(actual_outcomes)
        
        if self.method == "platt_scaling":
            # Platt缩放（逻辑回归）
            self.calibration_model = LogisticRegression()
            self.calibration_model.fit(predicted_confidences, actual_outcomes)
            
        elif self.method == "isotonic":
            # 保序回归
            from sklearn.isotonic import IsotonicRegression
            self.calibration_model = IsotonicRegression(out_of_bounds='clip')
            self.calibration_model.fit(predicted_confidences.ravel(), actual_outcomes)
        
        self.is_fitted = True
        
        # 计算校准指标
        calibrated_probs = self.predict(predicted_confidences)
        self.calibration_metrics = self._calculate_calibration_metrics(
            predicted_confidences.ravel(), actual_outcomes, calibrated_probs
        )
        
        return self
    
    def predict(self, predicted_confidences: np.ndarray) -> np.ndarray:
        """预测校准后的置信度"""
        if not self.is_fitted:
            raise ValueError("校准模型未训练")
        
        predicted_confidences = np.array(predicted_confidences).reshape(-1, 1)
        calibrated_probs = self.calibration_model.predict_proba(predicted_confidences)[:, 1]
        
        return calibrated_probs
    
    def _calculate_calibration_metrics(self, 
                                     original_probs: np.ndarray,
                                     true_labels: np.ndarray,
                                     calibrated_probs: np.ndarray) -> Dict[str, float]:
        """计算校准指标"""
        # 期望校准误差 (ECE)
        ece = self._calculate_ece(calibrated_probs, true_labels)
        
        # 最大校准误差 (MCE)
        mce = self._calculate_mce(calibrated_probs, true_labels)
        
        # Brier分数
        brier_score = np.mean((calibrated_probs - true_labels) ** 2)
        
        # 可靠性图面积
        reliability_score = 1.0 - ece
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score,
            'reliability_score': reliability_score
        }
    
    def _calculate_ece(self, probabilities: np.ndarray, true_labels: np.ndarray, n_bins: int = 10) -> float:
        """计算期望校准误差"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(self, probabilities: np.ndarray, true_labels: np.ndarray, n_bins: int = 10) -> float:
        """计算最大校准误差"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error

class DynamicConfidenceAdjuster:
    """动态置信度调整器"""
    
    def __init__(self, adaptation_rate: float = 0.1):
        self.adaptation_rate = adaptation_rate
        self.historical_performance = deque(maxlen=1000)
        self.current_adjustment = 1.0
        self.performance_trend = []
        
    def update_performance(self, 
                          predicted_confidence: float,
                          actual_outcome: float,
                          timestamp: datetime) -> None:
        """更新性能历史"""
        performance_record = {
            'predicted_confidence': predicted_confidence,
            'actual_outcome': actual_outcome,
            'timestamp': timestamp,
            'error': abs(predicted_confidence - actual_outcome)
        }
        
        self.historical_performance.append(performance_record)
        
        # 更新性能趋势
        if len(self.historical_performance) >= 10:
            recent_errors = [record['error'] for record in list(self.historical_performance)[-10:]]
            avg_error = np.mean(recent_errors)
            self.performance_trend.append(avg_error)
    
    def calculate_adjustment_factor(self) -> float:
        """计算调整因子"""
        if len(self.historical_performance) < 10:
            return 1.0
        
        # 计算近期性能
        recent_records = list(self.historical_performance)[-50:]
        recent_errors = [record['error'] for record in recent_records]
        recent_avg_error = np.mean(recent_errors)
        
        # 计算历史性能
        all_errors = [record['error'] for record in self.historical_performance]
        historical_avg_error = np.mean(all_errors)
        
        # 基于性能变化计算调整因子
        if recent_avg_error > historical_avg_error * 1.1:  # 性能下降
            adjustment_factor = 1.0 - self.adaptation_rate
        elif recent_avg_error < historical_avg_error * 0.9:  # 性能提升
            adjustment_factor = 1.0 + self.adaptation_rate
        else:
            adjustment_factor = 1.0
        
        # 平滑调整
        self.current_adjustment = (0.8 * self.current_adjustment + 
                                 0.2 * adjustment_factor)
        
        return self.current_adjustment
    
    def adjust_confidence(self, base_confidence: float) -> float:
        """调整置信度"""
        adjustment_factor = self.calculate_adjustment_factor()
        adjusted_confidence = base_confidence * adjustment_factor
        
        return np.clip(adjusted_confidence, 0.0, 1.0)
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """获取性能洞察"""
        if len(self.historical_performance) < 10:
            return {"status": "insufficient_data"}
        
        recent_errors = [record['error'] for record in list(self.historical_performance)[-50:]]
        all_errors = [record['error'] for record in self.historical_performance]
        
        insights = {
            'recent_avg_error': np.mean(recent_errors),
            'historical_avg_error': np.mean(all_errors),
            'error_trend': 'improving' if np.mean(recent_errors) < np.mean(all_errors) else 'degrading',
            'current_adjustment': self.current_adjustment,
            'total_records': len(self.historical_performance)
        }
        
        return insights

class ConfidenceAlertSystem:
    """置信度预警系统"""
    
    def __init__(self, alert_thresholds: Dict[str, float] = None):
        self.alert_thresholds = alert_thresholds or {
            'low_confidence': 0.3,
            'high_uncertainty': 0.7,
            'model_degradation': 0.1,
            'calibration_drift': 0.05
        }
        self.alert_history = deque(maxlen=1000)
        self.active_alerts = {}
        
    def check_alerts(self, 
                    confidence_metrics: ConfidenceMetrics,
                    uncertainty_metrics: Dict[str, float],
                    calibration_metrics: Dict[str, float]) -> List[AlertInfo]:
        """检查预警条件"""
        alerts = []
        
        # 低置信度预警
        if confidence_metrics.overall_confidence < self.alert_thresholds['low_confidence']:
            alert = AlertInfo(
                alert_type=AlertType.LOW_CONFIDENCE,
                severity='high',
                message=f"整体置信度过低: {confidence_metrics.overall_confidence:.3f}",
                confidence_score=confidence_metrics.overall_confidence,
                recommendations=self._get_low_confidence_recommendations(confidence_metrics)
            )
            alerts.append(alert)
        
        # 高不确定性预警
        total_uncertainty = uncertainty_metrics.get('total_uncertainty', 0)
        if total_uncertainty > self.alert_thresholds['high_uncertainty']:
            alert = AlertInfo(
                alert_type=AlertType.HIGH_UNCERTAINTY,
                severity='medium',
                message=f"不确定性过高: {total_uncertainty:.3f}",
                confidence_score=1.0 - total_uncertainty,
                recommendations=self._get_uncertainty_recommendations(uncertainty_metrics)
            )
            alerts.append(alert)
        
        # 模型退化预警
        ece = calibration_metrics.get('ece', 0)
        if ece > self.alert_thresholds['model_degradation']:
            alert = AlertInfo(
                alert_type=AlertType.MODEL_DEGRADATION,
                severity='medium',
                message=f"模型校准误差过大: ECE = {ece:.3f}",
                confidence_score=1.0 - ece,
                recommendations=["重新训练模型", "增加训练数据", "调整模型参数"]
            )
            alerts.append(alert)
        
        # 校准漂移预警
        reliability_score = calibration_metrics.get('reliability_score', 1.0)
        if reliability_score < self.alert_thresholds['calibration_drift']:
            alert = AlertInfo(
                alert_type=AlertType.CALIBRATION_DRIFT,
                severity='low',
                message=f"置信度校准漂移: 可靠性 = {reliability_score:.3f}",
                confidence_score=reliability_score,
                recommendations=["重新校准模型", "更新校准数据", "监控数据分布变化"]
            )
            alerts.append(alert)
        
        # 更新预警历史
        for alert in alerts:
            self.alert_history.append(alert)
            self.active_alerts[alert.alert_type] = alert
        
        return alerts
    
    def _get_low_confidence_recommendations(self, 
                                          metrics: ConfidenceMetrics) -> List[str]:
        """获取低置信度建议"""
        recommendations = []
        
        if metrics.prediction_confidence < 0.5:
            recommendations.append("增加更多训练样本")
            recommendations.append("优化特征工程")
        
        if metrics.model_confidence < 0.5:
            recommendations.append("尝试不同的模型算法")
            recommendations.append("调整模型超参数")
        
        if metrics.data_confidence < 0.5:
            recommendations.append("改善数据质量")
            recommendations.append("增加数据预处理步骤")
        
        if metrics.temporal_confidence < 0.5:
            recommendations.append("考虑时间序列特性")
            recommendations.append("调整预测时间窗口")
        
        return recommendations
    
    def _get_uncertainty_recommendations(self, 
                                       uncertainty_metrics: Dict[str, float]) -> List[str]:
        """获取不确定性建议"""
        recommendations = []
        
        aleatoric_uncertainty = uncertainty_metrics.get('aleatoric_uncertainty', 0)
        epistemic_uncertainty = uncertainty_metrics.get('epistemic_uncertainty', 0)
        
        if aleatoric_uncertainty > epistemic_uncertainty:
            recommendations.append("数据噪声较高，考虑数据清洗")
            recommendations.append("增加数据收集频率")
        else:
            recommendations.append("模型不确定性较高，考虑集成学习")
            recommendations.append("增加模型复杂度")
        
        return recommendations
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """获取预警摘要"""
        if not self.alert_history:
            return {"status": "no_alerts"}
        
        recent_alerts = list(self.alert_history)[-10:]
        alert_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for alert in recent_alerts:
            alert_counts[alert.alert_type.value] += 1
            severity_counts[alert.severity] += 1
        
        return {
            'total_alerts': len(self.alert_history),
            'recent_alert_count': len(recent_alerts),
            'alert_type_distribution': dict(alert_counts),
            'severity_distribution': dict(severity_counts),
            'active_alerts': len(self.active_alerts)
        }

class ConfidenceOptimizer:
    """置信度优化器"""
    
    def __init__(self):
        self.optimization_strategies = {
            'feature_selection': self._optimize_features,
            'model_selection': self._optimize_model,
            'data_enhancement': self._optimize_data,
            'ensemble_methods': self._optimize_ensemble
        }
        
    def generate_optimization_suggestions(self, 
                                        confidence_metrics: ConfidenceMetrics,
                                        uncertainty_metrics: Dict[str, float],
                                        performance_history: List[Dict]) -> Dict[str, List[str]]:
        """生成优化建议"""
        suggestions = {}
        
        # 基于置信度组件的建议
        if confidence_metrics.prediction_confidence < 0.6:
            suggestions['prediction_improvement'] = [
                "增加更多历史数据进行训练",
                "使用更复杂的预测模型",
                "考虑集成多个预测模型",
                "优化预测时间窗口"
            ]
        
        if confidence_metrics.model_confidence < 0.6:
            suggestions['model_improvement'] = [
                "尝试不同的机器学习算法",
                "进行超参数调优",
                "增加交叉验证",
                "使用正则化技术"
            ]
        
        if confidence_metrics.data_confidence < 0.6:
            suggestions['data_improvement'] = [
                "改善数据收集流程",
                "增加数据验证步骤",
                "处理缺失值和异常值",
                "增加数据多样性"
            ]
        
        # 基于不确定性的建议
        total_uncertainty = uncertainty_metrics.get('total_uncertainty', 0)
        if total_uncertainty > 0.5:
            suggestions['uncertainty_reduction'] = [
                "增加训练样本数量",
                "使用更稳定的特征",
                "考虑贝叶斯方法",
                "实施主动学习策略"
            ]
        
        # 基于性能历史的建议
        if len(performance_history) > 10:
            recent_performance = np.mean([p.get('accuracy', 0) for p in performance_history[-10:]])
            historical_performance = np.mean([p.get('accuracy', 0) for p in performance_history])
            
            if recent_performance < historical_performance * 0.9:
                suggestions['performance_recovery'] = [
                    "重新评估数据分布变化",
                    "更新模型训练策略",
                    "增加模型监控频率",
                    "考虑在线学习更新"
                ]
        
        return suggestions
    
    def _optimize_features(self, X: np.ndarray, y: np.ndarray) -> List[str]:
        """特征优化建议"""
        suggestions = []
        
        # 特征重要性分析
        if X.shape[1] > 20:
            suggestions.append("考虑特征选择以减少维度")
        
        if X.shape[0] < X.shape[1] * 5:
            suggestions.append("增加样本数量或减少特征数量")
        
        # 特征质量检查
        feature_missing_rates = np.mean(np.isnan(X), axis=0)
        if np.any(feature_missing_rates > 0.1):
            suggestions.append("处理高缺失率特征")
        
        return suggestions
    
    def _optimize_model(self, X: np.ndarray, y: np.ndarray) -> List[str]:
        """模型优化建议"""
        suggestions = []
        
        # 模型复杂度建议
        if X.shape[0] < 1000:
            suggestions.append("使用较简单的模型避免过拟合")
        elif X.shape[0] > 10000:
            suggestions.append("可以考虑更复杂的模型")
        
        # 目标变量分析
        if len(np.unique(y)) < 5:
            suggestions.append("考虑分类方法而非回归")
        
        return suggestions
    
    def _optimize_data(self, X: np.ndarray, y: np.ndarray) -> List[str]:
        """数据优化建议"""
        suggestions = []
        
        # 数据量检查
        if X.shape[0] < 100:
            suggestions.append("严重数据不足，需要大量增加样本")
        elif X.shape[0] < 1000:
            suggestions.append("数据量偏少，建议增加样本")
        
        # 类别平衡检查
        if len(np.unique(y)) > 2:
            unique, counts = np.unique(y, return_counts=True)
            if np.min(counts) / np.max(counts) < 0.1:
                suggestions.append("目标变量分布不均衡，考虑重采样")
        
        return suggestions
    
    def _optimize_ensemble(self, base_models: List[Any]) -> List[str]:
        """集成优化建议"""
        suggestions = []
        
        if len(base_models) < 3:
            suggestions.append("考虑使用更多基础模型进行集成")
        
        suggestions.append("尝试不同的集成策略（投票、平均、堆叠）")
        suggestions.append("考虑使用贝叶斯模型平均")
        
        return suggestions

class ConfidenceAssessor:
    """D5置信度评估器主类"""
    
    def __init__(self, 
                 model_type: str = "random_forest",
                 calibration_method: str = "platt_scaling",
                 adaptation_rate: float = 0.1):
        """
        初始化置信度评估器
        
        Args:
            model_type: 基础模型类型
            calibration_method: 校准方法
            adaptation_rate: 自适应调整率
        """
        self.confidence_model = ConfidenceModel(model_type)
        self.confidence_calculator = MultiDimensionalConfidenceCalculator()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.calibrator = ConfidenceCalibrator(calibration_method)
        self.dynamic_adjuster = DynamicConfidenceAdjuster(adaptation_rate)
        self.alert_system = ConfidenceAlertSystem()
        self.optimizer = ConfidenceOptimizer()
        
        # 状态跟踪
        self.is_initialized = False
        self.training_history = []
        self.performance_cache = {}
        
    def initialize(self, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None,
                   data_quality_metrics: Optional[Dict[str, float]] = None) -> None:
        """初始化评估器"""
        logger.info("初始化置信度评估器...")
        
        # 训练基础模型
        self.confidence_model.fit(X_train, y_train)
        
        # 计算初始置信度组件
        if X_val is not None and y_val is not None:
            val_predictions = self.confidence_model.predict(X_val)
            
            # 计算各个维度的置信度
            prediction_confidence = self.confidence_calculator.calculate_prediction_confidence(
                val_predictions, y_val
            )
            
            model_confidence = self.confidence_calculator.calculate_model_confidence(
                self.confidence_model, X_val, y_val
            )
            
            data_confidence = self.confidence_calculator.calculate_data_confidence(
                X_train, data_quality_metrics or {}
            )
            
            # 存储初始状态
            initial_metrics = ConfidenceMetrics(
                overall_confidence=0.0,
                prediction_confidence=prediction_confidence,
                model_confidence=model_confidence,
                data_confidence=data_confidence,
                temporal_confidence=0.5,  # 默认值
                uncertainty=0.0,
                confidence_interval=(0.0, 1.0),
                confidence_level=ConfidenceLevel.MEDIUM
            )
            
            self.performance_cache['initial_metrics'] = initial_metrics
        
        self.is_initialized = True
        logger.info("置信度评估器初始化完成")
    
    def assess_confidence(self, 
                         X: np.ndarray,
                         y_true: Optional[np.ndarray] = None,
                         timestamps: Optional[List[datetime]] = None,
                         data_quality_metrics: Optional[Dict[str, float]] = None) -> ConfidenceMetrics:
        """评估置信度"""
        if not self.is_initialized:
            raise ValueError("评估器未初始化")
        
        # 获取预测结果
        predictions = self.confidence_model.predict(X)
        
        # 计算各维度置信度
        prediction_confidence = self.confidence_calculator.calculate_prediction_confidence(
            predictions, y_true
        )
        
        model_confidence = self.confidence_calculator.calculate_model_confidence(
            self.confidence_model, X, y_true if y_true is not None else predictions
        )
        
        data_confidence = self.confidence_calculator.calculate_data_confidence(
            X, data_quality_metrics or {}
        )
        
        temporal_confidence = self.confidence_calculator.calculate_temporal_confidence(
            timestamps or [datetime.now()]
        )
        
        # 计算综合置信度
        confidence_components = {
            'prediction': prediction_confidence,
            'model': model_confidence,
            'data': data_confidence,
            'temporal': temporal_confidence
        }
        
        overall_confidence = self.confidence_calculator.calculate_overall_confidence(
            confidence_components
        )
        
        # 计算不确定性
        uncertainty_metrics = self.uncertainty_quantifier.calculate_bootstrap_uncertainty(predictions)
        total_uncertainty = uncertainty_metrics.get('std', 0.0)
        
        # 计算置信区间
        confidence_interval = self.uncertainty_quantifier.calculate_prediction_interval(predictions)
        
        # 动态调整置信度
        if y_true is not None:
            for pred, actual in zip(predictions, y_true):
                self.dynamic_adjuster.update_performance(pred, actual, datetime.now())
            
            overall_confidence = self.dynamic_adjuster.adjust_confidence(overall_confidence)
        
        # 确定置信度等级
        if overall_confidence >= 0.8:
            confidence_level = ConfidenceLevel.VERY_HIGH
        elif overall_confidence >= 0.6:
            confidence_level = ConfidenceLevel.HIGH
        elif overall_confidence >= 0.4:
            confidence_level = ConfidenceLevel.MEDIUM
        elif overall_confidence >= 0.2:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.VERY_LOW
        
        # 构建置信度指标
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=overall_confidence,
            prediction_confidence=prediction_confidence,
            model_confidence=model_confidence,
            data_confidence=data_confidence,
            temporal_confidence=temporal_confidence,
            uncertainty=total_uncertainty,
            confidence_interval=confidence_interval,
            confidence_level=confidence_level,
            metadata={
                'uncertainty_metrics': uncertainty_metrics,
                'prediction_stats': {
                    'mean': float(np.mean(predictions)),
                    'std': float(np.std(predictions)),
                    'min': float(np.min(predictions)),
                    'max': float(np.max(predictions))
                }
            }
        )
        
        return confidence_metrics
    
    def calibrate_confidence(self, 
                           predicted_confidences: np.ndarray,
                           actual_outcomes: np.ndarray) -> ConfidenceCalibrator:
        """校准置信度"""
        logger.info("开始置信度校准...")
        
        self.calibrator.fit(predicted_confidences, actual_outcomes)
        
        logger.info(f"校准完成，ECE: {self.calibrator.calibration_metrics.get('ece', 0):.4f}")
        
        return self.calibrator
    
    def check_alerts(self, confidence_metrics: ConfidenceMetrics) -> List[AlertInfo]:
        """检查预警"""
        uncertainty_metrics = confidence_metrics.metadata.get('uncertainty_metrics', {})
        calibration_metrics = self.calibrator.calibration_metrics
        
        alerts = self.alert_system.check_alerts(
            confidence_metrics, uncertainty_metrics, calibration_metrics
        )
        
        return alerts
    
    def get_optimization_suggestions(self, 
                                   confidence_metrics: ConfidenceMetrics) -> Dict[str, List[str]]:
        """获取优化建议"""
        uncertainty_metrics = confidence_metrics.metadata.get('uncertainty_metrics', {})
        
        suggestions = self.optimizer.generate_optimization_suggestions(
            confidence_metrics, uncertainty_metrics, self.training_history
        )
        
        return suggestions
    
    def get_detailed_report(self, confidence_metrics: ConfidenceMetrics) -> Dict[str, Any]:
        """获取详细报告"""
        # 基础信息
        report = {
            'timestamp': confidence_metrics.timestamp.isoformat(),
            'overall_confidence': confidence_metrics.overall_confidence,
            'confidence_level': confidence_metrics.confidence_level.value,
            'confidence_components': {
                'prediction': confidence_metrics.prediction_confidence,
                'model': confidence_metrics.model_confidence,
                'data': confidence_metrics.data_confidence,
                'temporal': confidence_metrics.temporal_confidence
            },
            'uncertainty_analysis': {
                'total_uncertainty': confidence_metrics.uncertainty,
                'confidence_interval': confidence_metrics.confidence_interval,
                'prediction_stats': confidence_metrics.metadata.get('prediction_stats', {})
            }
        }
        
        # 预警信息
        alerts = self.check_alerts(confidence_metrics)
        if alerts:
            report['alerts'] = [
                {
                    'type': alert.alert_type.value,
                    'severity': alert.severity,
                    'message': alert.message,
                    'recommendations': alert.recommendations
                }
                for alert in alerts
            ]
        
        # 优化建议
        suggestions = self.get_optimization_suggestions(confidence_metrics)
        if suggestions:
            report['optimization_suggestions'] = suggestions
        
        # 性能洞察
        performance_insights = self.dynamic_adjuster.get_performance_insights()
        if performance_insights:
            report['performance_insights'] = performance_insights
        
        # 校准信息
        if self.calibrator.calibration_metrics:
            report['calibration_metrics'] = self.calibrator.calibration_metrics
        
        return report
    
    def save_state(self, filepath: str) -> None:
        """保存状态"""
        state = {
            'confidence_model': self.confidence_model,
            'calibration_metrics': self.calibrator.calibration_metrics,
            'performance_cache': self.performance_cache,
            'training_history': self.training_history,
            'is_initialized': self.is_initialized
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"状态已保存到: {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """加载状态"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.confidence_model = state['confidence_model']
        self.calibrator.calibration_metrics = state['calibration_metrics']
        self.performance_cache = state['performance_cache']
        self.training_history = state['training_history']
        self.is_initialized = state['is_initialized']
        
        logger.info(f"状态已从 {filepath} 加载")

# 使用示例和测试代码
def demo_confidence_assessor():
    """演示置信度评估器的使用"""
    print("=== D5置信度评估器演示 ===\n")
    
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.1
    
    # 分割训练和测试数据
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 数据质量指标
    data_quality_metrics = {
        'completeness': 0.95,
        'consistency': 0.90,
        'accuracy': 0.85
    }
    
    # 初始化置信度评估器
    assessor = ConfidenceAssessor(model_type="random_forest")
    
    print("1. 初始化评估器...")
    assessor.initialize(X_train, y_train, X_test, y_test, data_quality_metrics)
    
    print("2. 评估测试数据置信度...")
    timestamps = [datetime.now() + timedelta(days=i) for i in range(len(X_test))]
    confidence_metrics = assessor.assess_confidence(X_test, y_test, timestamps, data_quality_metrics)
    
    print(f"   整体置信度: {confidence_metrics.overall_confidence:.4f}")
    print(f"   置信度等级: {confidence_metrics.confidence_level.value}")
    print(f"   预测置信度: {confidence_metrics.prediction_confidence:.4f}")
    print(f"   模型置信度: {confidence_metrics.model_confidence:.4f}")
    print(f"   数据置信度: {confidence_metrics.data_confidence:.4f}")
    print(f"   时间置信度: {confidence_metrics.temporal_confidence:.4f}")
    print(f"   不确定性: {confidence_metrics.uncertainty:.4f}")
    print(f"   置信区间: {confidence_metrics.confidence_interval}")
    
    print("\n3. 置信度校准...")
    # 模拟置信度校准
    predicted_confidences = np.random.beta(2, 2, 100)
    actual_outcomes = np.random.binomial(1, predicted_confidences)
    calibrator = assessor.calibrate_confidence(predicted_confidences, actual_outcomes)
    print(f"   校准后ECE: {calibrator.calibration_metrics.get('ece', 0):.4f}")
    
    print("\n4. 检查预警...")
    alerts = assessor.check_alerts(confidence_metrics)
    if alerts:
        print(f"   发现 {len(alerts)} 个预警:")
        for alert in alerts:
            print(f"   - [{alert.severity.upper()}] {alert.alert_type.value}: {alert.message}")
            if alert.recommendations:
                print(f"     建议: {', '.join(alert.recommendations[:2])}")
    else:
        print("   未发现预警")
    
    print("\n5. 获取优化建议...")
    suggestions = assessor.get_optimization_suggestions(confidence_metrics)
    if suggestions:
        print("   优化建议:")
        for category, advice_list in suggestions.items():
            print(f"   - {category}: {', '.join(advice_list[:2])}")
    
    print("\n6. 生成详细报告...")
    report = assessor.get_detailed_report(confidence_metrics)
    print(f"   报告包含 {len(report)} 个主要部分")
    
    # 保存状态
    print("\n7. 保存评估器状态...")
    assessor.save_state('/tmp/confidence_assessor_state.pkl')
    print("   状态保存完成")
    
    print("\n=== 演示完成 ===")

if __name__ == "__main__":
    demo_confidence_assessor()