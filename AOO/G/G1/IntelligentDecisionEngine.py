#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G1智能决策引擎
实现多维度决策分析、模型构建、不确定性处理、风险评估、效果预测、学习改进和结果解释
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from scipy import stats
from scipy.optimize import minimize
import warnings
import logging
from datetime import datetime, timedelta
import json
import pickle
from collections import defaultdict, deque

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DecisionCriteria:
    """决策标准定义"""
    name: str
    weight: float
    type: str  # 'benefit' 或 'cost'
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None

@dataclass
class DecisionOption:
    """决策选项定义"""
    id: str
    name: str
    description: str = ""
    criteria_values: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecisionResult:
    """决策结果"""
    option_id: str
    score: float
    confidence: float
    risk_level: float
    uncertainty: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskAssessment:
    """风险评估结果"""
    risk_type: str
    probability: float
    impact: float
    risk_score: float
    mitigation_strategy: str
    description: str

class UncertaintyHandler:
    """不确定性处理器"""
    
    def __init__(self):
        self.uncertainty_models = {}
        
    def quantify_uncertainty(self, data: np.ndarray, method: str = 'bootstrap') -> Dict[str, float]:
        """量化数据不确定性"""
        if method == 'bootstrap':
            return self._bootstrap_uncertainty(data)
        elif method == 'bayesian':
            return self._bayesian_uncertainty(data)
        elif method == 'monte_carlo':
            return self._monte_carlo_uncertainty(data)
        else:
            raise ValueError(f"未知的不确定性量化方法: {method}")
    
    def _bootstrap_uncertainty(self, data: np.ndarray, n_bootstrap: int = 1000) -> Dict[str, float]:
        """Bootstrap方法量化不确定性"""
        n = len(data)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        return {
            'mean': np.mean(bootstrap_means),
            'std': np.std(bootstrap_means),
            'confidence_interval_95': np.percentile(bootstrap_means, [2.5, 97.5]),
            'coefficient_of_variation': np.std(bootstrap_means) / np.mean(bootstrap_means)
        }
    
    def _bayesian_uncertainty(self, data: np.ndarray) -> Dict[str, float]:
        """贝叶斯方法量化不确定性"""
        # 假设数据服从正态分布，使用共轭先验
        n = len(data)
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)
        
        # 无信息先验
        prior_alpha = 0.5
        prior_beta = 0.5
        
        # 后验参数
        posterior_alpha = prior_alpha + n / 2
        posterior_beta = prior_beta + (n * sample_var + n * (sample_mean ** 2)) / 2
        
        return {
            'posterior_mean': sample_mean,
            'posterior_variance': sample_var / n,
            'credible_interval_95': stats.t.interval(0.95, n-1, loc=sample_mean, scale=np.sqrt(sample_var/n)),
            'uncertainty_coefficient': np.sqrt(sample_var) / abs(sample_mean) if sample_mean != 0 else float('inf')
        }
    
    def _monte_carlo_uncertainty(self, data: np.ndarray, n_simulations: int = 10000) -> Dict[str, float]:
        """蒙特卡洛方法量化不确定性"""
        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        
        # 生成模拟数据
        simulated_means = []
        for _ in range(n_simulations):
            simulated_data = np.random.normal(sample_mean, sample_std, n)
            simulated_means.append(np.mean(simulated_data))
        
        return {
            'simulation_mean': np.mean(simulated_means),
            'simulation_std': np.std(simulated_means),
            'confidence_interval_95': np.percentile(simulated_means, [2.5, 97.5]),
            'risk_of_extreme': np.sum(np.abs(simulated_means) > 2 * sample_std) / n_simulations
        }

class RiskAssessmentEngine:
    """风险评估引擎"""
    
    def __init__(self):
        self.risk_models = {}
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
    
    def assess_risk(self, option: DecisionOption, context: Dict[str, Any] = None) -> List[RiskAssessment]:
        """评估决策风险"""
        risks = []
        
        # 技术风险
        tech_risk = self._assess_technical_risk(option)
        if tech_risk:
            risks.append(tech_risk)
        
        # 市场风险
        market_risk = self._assess_market_risk(option, context)
        if market_risk:
            risks.append(market_risk)
        
        # 财务风险
        financial_risk = self._assess_financial_risk(option)
        if financial_risk:
            risks.append(financial_risk)
        
        # 操作风险
        operational_risk = self._assess_operational_risk(option)
        if operational_risk:
            risks.append(operational_risk)
        
        return risks
    
    def _assess_technical_risk(self, option: DecisionOption) -> Optional[RiskAssessment]:
        """评估技术风险"""
        tech_complexity = option.metadata.get('technical_complexity', 0.5)
        tech_maturity = option.metadata.get('technology_maturity', 0.5)
        
        probability = tech_complexity * (1 - tech_maturity)
        impact = tech_complexity * 0.8
        risk_score = probability * impact
        
        if risk_score > 0.3:
            return RiskAssessment(
                risk_type="技术风险",
                probability=probability,
                impact=impact,
                risk_score=risk_score,
                mitigation_strategy="技术预研、原型验证、专家咨询",
                description=f"技术复杂度: {tech_complexity:.2f}, 技术成熟度: {tech_maturity:.2f}"
            )
        return None
    
    def _assess_market_risk(self, option: DecisionOption, context: Dict[str, Any] = None) -> Optional[RiskAssessment]:
        """评估市场风险"""
        market_volatility = context.get('market_volatility', 0.5) if context else 0.5
        competition_intensity = context.get('competition_intensity', 0.5) if context else 0.5
        
        probability = market_volatility * competition_intensity
        impact = competition_intensity * 0.7
        risk_score = probability * impact
        
        if risk_score > 0.3:
            return RiskAssessment(
                risk_type="市场风险",
                probability=probability,
                impact=impact,
                risk_score=risk_score,
                mitigation_strategy="市场调研、竞争分析、灵活定价策略",
                description=f"市场波动性: {market_volatility:.2f}, 竞争强度: {competition_intensity:.2f}"
            )
        return None
    
    def _assess_financial_risk(self, option: DecisionOption) -> Optional[RiskAssessment]:
        """评估财务风险"""
        investment_size = option.metadata.get('investment_size', 0.5)
        roi_uncertainty = option.metadata.get('roi_uncertainty', 0.5)
        
        probability = investment_size * roi_uncertainty
        impact = investment_size * 0.9
        risk_score = probability * impact
        
        if risk_score > 0.3:
            return RiskAssessment(
                risk_type="财务风险",
                probability=probability,
                impact=impact,
                risk_score=risk_score,
                mitigation_strategy="分阶段投资、财务监控、风险对冲",
                description=f"投资规模: {investment_size:.2f}, ROI不确定性: {roi_uncertainty:.2f}"
            )
        return None
    
    def _assess_operational_risk(self, option: DecisionOption) -> Optional[RiskAssessment]:
        """评估操作风险"""
        operational_complexity = option.metadata.get('operational_complexity', 0.5)
        resource_availability = option.metadata.get('resource_availability', 0.5)
        
        probability = operational_complexity * (1 - resource_availability)
        impact = operational_complexity * 0.6
        risk_score = probability * impact
        
        if risk_score > 0.3:
            return RiskAssessment(
                risk_type="操作风险",
                probability=probability,
                impact=impact,
                risk_score=risk_score,
                mitigation_strategy="流程优化、人员培训、资源保障",
                description=f"操作复杂度: {operational_complexity:.2f}, 资源可用性: {resource_availability:.2f}"
            )
        return None

class EffectPredictionModel:
    """决策效果预测模型"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def train_prediction_models(self, historical_data: pd.DataFrame, target_column: str):
        """训练预测模型"""
        X = historical_data.drop(columns=[target_column])
        y = historical_data[target_column]
        
        # 数据预处理
        X_scaled = self._preprocess_data(X)
        
        # 训练多种模型
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        model_scores = {}
        
        for name, model in models.items():
            try:
                # 训练模型
                model.fit(X_scaled, y)
                
                # 交叉验证评估
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                model_scores[name] = np.mean(cv_scores)
                
                # 保存模型和缩放器
                self.models[name] = model
                self.scalers[name] = StandardScaler()
                self.scalers[name].fit(X)
                
                # 计算特征重要性
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = dict(zip(X.columns, np.abs(model.coef_)))
                
                logger.info(f"{name} 模型训练完成，R²分数: {np.mean(cv_scores):.3f}")
                
            except Exception as e:
                logger.warning(f"{name} 模型训练失败: {str(e)}")
        
        return model_scores
    
    def predict_effect(self, option: DecisionOption, model_name: str = 'best') -> Dict[str, float]:
        """预测决策效果"""
        if model_name == 'best':
            model_name = self._select_best_model()
        
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 未训练")
        
        # 准备特征数据
        features = self._option_to_features(option)
        features_scaled = self.scalers[model_name].transform([features])
        
        # 预测
        prediction = self.models[model_name].predict(features_scaled)[0]
        
        # 计算置信区间
        confidence_interval = self._calculate_confidence_interval(features_scaled, model_name)
        
        return {
            'predicted_effect': prediction,
            'confidence_interval': confidence_interval,
            'model_used': model_name,
            'feature_importance': self.feature_importance.get(model_name, {})
        }
    
    def _preprocess_data(self, X: pd.DataFrame) -> np.ndarray:
        """数据预处理"""
        # 处理分类变量
        X_processed = X.copy()
        
        for column in X_processed.columns:
            if X_processed[column].dtype == 'object':
                le = LabelEncoder()
                X_processed[column] = le.fit_transform(X_processed[column].astype(str))
        
        return X_processed.values
    
    def _option_to_features(self, option: DecisionOption) -> List[float]:
        """将决策选项转换为特征向量"""
        features = []
        
        # 基础特征
        features.append(len(option.criteria_values))
        features.append(option.metadata.get('complexity', 0.5))
        features.append(option.metadata.get('cost', 0.5))
        features.append(option.metadata.get('time', 0.5))
        
        # 标准值特征
        for criteria_name in ['cost', 'benefit', 'risk', 'time', 'quality']:
            features.append(option.criteria_values.get(criteria_name, 0.5))
        
        return features
    
    def _select_best_model(self) -> str:
        """选择最佳模型"""
        if not self.models:
            raise ValueError("没有可用的模型")
        
        # 这里可以根据交叉验证分数选择最佳模型
        # 简化处理，返回第一个模型
        return list(self.models.keys())[0]
    
    def _calculate_confidence_interval(self, features: np.ndarray, model_name: str) -> Tuple[float, float]:
        """计算置信区间"""
        # 简化的置信区间计算
        prediction = self.models[model_name].predict(features)[0]
        uncertainty = 0.1  # 假设10%的不确定性
        
        lower = prediction * (1 - uncertainty)
        upper = prediction * (1 + uncertainty)
        
        return (lower, upper)

class LearningEngine:
    """决策学习引擎"""
    
    def __init__(self):
        self.learning_history = deque(maxlen=1000)
        self.model_performance = defaultdict(list)
        self.improvement_suggestions = []
        
    def learn_from_feedback(self, decision_id: str, actual_outcome: float, 
                          predicted_outcome: float, context: Dict[str, Any]):
        """从反馈中学习"""
        # 计算预测误差
        error = abs(actual_outcome - predicted_outcome)
        relative_error = error / abs(actual_outcome) if actual_outcome != 0 else error
        
        # 记录学习历史
        learning_record = {
            'decision_id': decision_id,
            'timestamp': datetime.now(),
            'actual_outcome': actual_outcome,
            'predicted_outcome': predicted_outcome,
            'error': error,
            'relative_error': relative_error,
            'context': context
        }
        
        self.learning_history.append(learning_record)
        
        # 更新模型性能
        self.model_performance['errors'].append(error)
        self.model_performance['relative_errors'].append(relative_error)
        
        # 生成改进建议
        self._generate_improvement_suggestions()
        
        logger.info(f"学习记录已更新，决策ID: {decision_id}, 相对误差: {relative_error:.3f}")
    
    def _generate_improvement_suggestions(self):
        """生成改进建议"""
        if len(self.model_performance['relative_errors']) < 10:
            return
        
        recent_errors = self.model_performance['relative_errors'][-10:]
        avg_error = np.mean(recent_errors)
        
        suggestions = []
        
        if avg_error > 0.2:
            suggestions.append("预测误差较高，建议增加训练数据或调整模型参数")
        
        if len(set(self.model_performance['relative_errors'][-5:])) < 3:
            suggestions.append("预测结果变化较小，可能存在模型过拟合，建议正则化")
        
        error_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
        if error_trend > 0.01:
            suggestions.append("预测误差呈上升趋势，建议重新训练模型")
        
        self.improvement_suggestions.extend(suggestions)
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """获取学习洞察"""
        if not self.learning_history:
            return {"message": "暂无学习历史"}
        
        recent_records = list(self.learning_history)[-20:]
        
        insights = {
            'total_decisions': len(self.learning_history),
            'recent_performance': {
                'avg_error': np.mean([r['error'] for r in recent_records]),
                'avg_relative_error': np.mean([r['relative_error'] for r in recent_records]),
                'error_std': np.std([r['error'] for r in recent_records])
            },
            'improvement_suggestions': self.improvement_suggestions[-5:],
            'trend_analysis': self._analyze_trend()
        }
        
        return insights
    
    def _analyze_trend(self) -> Dict[str, float]:
        """分析趋势"""
        if len(self.model_performance['relative_errors']) < 10:
            return {}
        
        errors = self.model_performance['relative_errors'][-20:]
        x = np.arange(len(errors))
        
        # 线性回归分析趋势
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, errors)
        
        return {
            'trend_slope': slope,
            'trend_r_squared': r_value ** 2,
            'trend_significance': p_value
        }

class ExplanationEngine:
    """决策解释引擎"""
    
    def __init__(self):
        self.explanation_templates = {
            'criteria_based': "基于{criteria_count}个决策标准，{option_name}获得了{score:.3f}的综合评分",
            'risk_aware': "考虑到{high_risk_count}个高风险因素，{option_name}的风险调整评分为{adjusted_score:.3f}",
            'uncertainty_aware': "在{uncertainty_level}不确定性水平下，{option_name}的置信区间为[{lower:.3f}, {upper:.3f}]",
            'comparison_based': "与最佳替代方案相比，{option_name}的优势在于{advantages}，劣势在于{disadvantages}"
        }
    
    def generate_explanation(self, result: DecisionResult, all_results: List[DecisionResult], 
                           criteria: List[DecisionCriteria]) -> str:
        """生成决策解释"""
        explanations = []
        
        # 基于标准的解释
        criteria_explanation = self._explain_by_criteria(result, criteria)
        if criteria_explanation:
            explanations.append(criteria_explanation)
        
        # 基于风险的解释
        risk_explanation = self._explain_by_risk(result)
        if risk_explanation:
            explanations.append(risk_explanation)
        
        # 基于不确定性的解释
        uncertainty_explanation = self._explain_by_uncertainty(result)
        if uncertainty_explanation:
            explanations.append(uncertainty_explanation)
        
        # 基于比较的解释
        comparison_explanation = self._explain_by_comparison(result, all_results)
        if comparison_explanation:
            explanations.append(comparison_explanation)
        
        return "\n".join(explanations)
    
    def _explain_by_criteria(self, result: DecisionResult, criteria: List[DecisionCriteria]) -> str:
        """基于决策标准解释"""
        if not result.metadata.get('criteria_scores'):
            return ""
        
        criteria_scores = result.metadata['criteria_scores']
        top_criteria = sorted(criteria_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        explanation = "决策分析基于以下关键标准:\n"
        for criteria_name, score in top_criteria:
            explanation += f"• {criteria_name}: {score:.3f}\n"
        
        return explanation
    
    def _explain_by_risk(self, result: DecisionResult) -> str:
        """基于风险解释"""
        if result.risk_level < 0.3:
            risk_level_desc = "低"
        elif result.risk_level < 0.6:
            risk_level_desc = "中等"
        else:
            risk_level_desc = "高"
        
        return f"该决策的风险水平为{risk_level_desc}({result.risk_level:.3f})，建议在实施前制定相应的风险缓解策略。"
    
    def _explain_by_uncertainty(self, result: DecisionResult) -> str:
        """基于不确定性解释"""
        if result.uncertainty < 0.2:
            uncertainty_level = "低"
        elif result.uncertainty < 0.5:
            uncertainty_level = "中等"
        else:
            uncertainty_level = "高"
        
        confidence_interval = result.metadata.get('confidence_interval', (0, 1))
        return f"决策预测存在{uncertainty_level}水平的不确定性，置信区间为[{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]。"
    
    def _explain_by_comparison(self, result: DecisionResult, all_results: List[DecisionResult]) -> str:
        """基于比较解释"""
        if len(all_results) < 2:
            return ""
        
        # 找到排名
        sorted_results = sorted(all_results, key=lambda x: x.score, reverse=True)
        rank = next(i for i, r in enumerate(sorted_results, 1) if r.option_id == result.option_id)
        
        if rank == 1:
            return "该决策在所有候选方案中排名第一，是当前最优选择。"
        elif rank <= len(sorted_results) // 3:
            return f"该决策在所有候选方案中排名第{rank}位，属于优秀选择。"
        else:
            return f"该决策在所有候选方案中排名第{rank}位，建议进一步优化或考虑其他方案。"

class IntelligentDecisionEngine:
    """G1智能决策引擎"""
    
    def __init__(self):
        self.uncertainty_handler = UncertaintyHandler()
        self.risk_engine = RiskAssessmentEngine()
        self.prediction_model = EffectPredictionModel()
        self.learning_engine = LearningEngine()
        self.explanation_engine = ExplanationEngine()
        
        self.criteria = []
        self.options = []
        self.decision_history = []
        
        logger.info("G1智能决策引擎初始化完成")
    
    def add_criteria(self, criteria: List[DecisionCriteria]):
        """添加决策标准"""
        self.criteria.extend(criteria)
        logger.info(f"添加了 {len(criteria)} 个决策标准")
    
    def add_options(self, options: List[DecisionOption]):
        """添加决策选项"""
        self.options.extend(options)
        logger.info(f"添加了 {len(options)} 个决策选项")
    
    def make_decision(self, context: Dict[str, Any] = None) -> DecisionResult:
        """执行智能决策"""
        if not self.criteria or not self.options:
            raise ValueError("缺少决策标准或决策选项")
        
        logger.info("开始智能决策分析...")
        
        # 1. 多维度决策分析
        scored_options = self._multi_criteria_analysis()
        
        # 2. 不确定性处理
        for option in scored_options:
            option.uncertainty = self._calculate_uncertainty(option)
        
        # 3. 风险评估
        risk_assessments = {}
        for option in scored_options:
            risks = self.risk_engine.assess_risk(option, context)
            option.risk_level = np.mean([risk.risk_score for risk in risks]) if risks else 0.1
            risk_assessments[option.id] = risks
        
        # 4. 效果预测
        predictions = {}
        for option in scored_options:
            try:
                prediction = self.prediction_model.predict_effect(option)
                predictions[option.id] = prediction
                option.metadata.update(prediction)
            except Exception as e:
                logger.warning(f"效果预测失败: {str(e)}")
                predictions[option.id] = {'predicted_effect': 0.5}
        
        # 5. 综合评分
        final_results = []
        for option in scored_options:
            # 综合评分 = 基础评分 * (1 - 风险调整) * (1 - 不确定性调整)
            risk_adjustment = option.risk_level * 0.3
            uncertainty_adjustment = option.uncertainty * 0.2
            
            final_score = option.score * (1 - risk_adjustment) * (1 - uncertainty_adjustment)
            confidence = 1 - option.uncertainty - option.risk_level
            
            # 生成解释
            result = DecisionResult(
                option_id=option.id,
                score=final_score,
                confidence=max(0, confidence),
                risk_level=option.risk_level,
                uncertainty=option.uncertainty,
                explanation="",
                metadata={
                    'base_score': option.score,
                    'risk_adjustment': risk_adjustment,
                    'uncertainty_adjustment': uncertainty_adjustment,
                    'criteria_scores': option.criteria_values,
                    'predictions': predictions.get(option.id, {}),
                    'risk_assessments': risk_assessments.get(option.id, [])
                }
            )
            
            final_results.append(result)
        
        # 6. 生成解释
        for result in final_results:
            result.explanation = self.explanation_engine.generate_explanation(
                result, final_results, self.criteria
            )
        
        # 7. 记录决策历史
        decision_record = {
            'timestamp': datetime.now(),
            'context': context,
            'results': final_results,
            'criteria': self.criteria,
            'options': self.options
        }
        self.decision_history.append(decision_record)
        
        # 返回最佳决策
        best_result = max(final_results, key=lambda x: x.score)
        logger.info(f"决策完成，最佳选择: {best_result.option_id}, 评分: {best_result.score:.3f}")
        
        return best_result
    
    def _multi_criteria_analysis(self) -> List[DecisionOption]:
        """多准则决策分析"""
        scored_options = []
        
        for option in self.options:
            total_score = 0
            total_weight = 0
            
            for criteria in self.criteria:
                if criteria.name in option.criteria_values:
                    value = option.criteria_values[criteria.name]
                    
                    # 标准化值到[0,1]区间
                    normalized_value = self._normalize_value(value, criteria)
                    
                    # 根据标准类型计算得分
                    if criteria.type == 'benefit':
                        score = normalized_value
                    else:  # cost
                        score = 1 - normalized_value
                    
                    weighted_score = score * criteria.weight
                    total_score += weighted_score
                    total_weight += criteria.weight
                    
                    # 存储标准得分
                    option.criteria_values[f"{criteria.name}_normalized"] = normalized_value
                    option.criteria_values[f"{criteria.name}_score"] = score
            
            # 计算最终得分
            if total_weight > 0:
                option.score = total_score / total_weight
            else:
                option.score = 0.5
            
            scored_options.append(option)
        
        return scored_options
    
    def _normalize_value(self, value: float, criteria: DecisionCriteria) -> float:
        """标准化值"""
        if criteria.min_value is not None and criteria.max_value is not None:
            if criteria.max_value == criteria.min_value:
                return 0.5
            return (value - criteria.min_value) / (criteria.max_value - criteria.min_value)
        else:
            # 默认假设值在[0,1]范围内
            return max(0, min(1, value))
    
    def _calculate_uncertainty(self, option: DecisionOption) -> float:
        """计算决策不确定性"""
        uncertainties = []
        
        # 基于标准值的不确定性
        criteria_values = [v for k, v in option.criteria_values.items() 
                          if not k.endswith('_normalized') and not k.endswith('_score')]
        if criteria_values:
            uncertainty_data = self.uncertainty_handler.quantify_uncertainty(
                np.array(criteria_values), method='bootstrap'
            )
            uncertainties.append(uncertainty_data['coefficient_of_variation'])
        
        # 基于元数据的不确定性
        metadata_values = list(option.metadata.values())
        if metadata_values and all(isinstance(v, (int, float)) for v in metadata_values):
            uncertainty_data = self.uncertainty_handler.quantify_uncertainty(
                np.array(metadata_values), method='monte_carlo'
            )
            uncertainties.append(uncertainty_data['risk_of_extreme'])
        
        return np.mean(uncertainties) if uncertainties else 0.2
    
    def train_prediction_models(self, historical_data: pd.DataFrame, target_column: str):
        """训练预测模型"""
        logger.info("开始训练预测模型...")
        scores = self.prediction_model.train_prediction_models(historical_data, target_column)
        logger.info(f"模型训练完成: {scores}")
        return scores
    
    def learn_from_feedback(self, decision_id: str, actual_outcome: float, 
                          context: Dict[str, Any] = None):
        """从反馈中学习"""
        # 查找对应的决策记录
        decision_record = None
        for record in reversed(self.decision_history):
            if any(r.option_id == decision_id for r in record['results']):
                decision_record = record
                break
        
        if not decision_record:
            logger.warning(f"未找到决策记录: {decision_id}")
            return
        
        # 获取预测结果
        result = next(r for r in decision_record['results'] if r.option_id == decision_id)
        predicted_outcome = result.score
        
        # 学习
        self.learning_engine.learn_from_feedback(
            decision_id, actual_outcome, predicted_outcome, context or {}
        )
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """获取学习洞察"""
        return self.learning_engine.get_learning_insights()
    
    def get_decision_explanation(self, decision_id: str) -> Optional[str]:
        """获取决策解释"""
        for record in reversed(self.decision_history):
            for result in record['results']:
                if result.option_id == decision_id:
                    return result.explanation
        return None
    
    def export_decision_report(self, decision_id: str, output_file: str):
        """导出决策报告"""
        # 查找决策记录
        decision_record = None
        for record in reversed(self.decision_history):
            if any(r.option_id == decision_id for r in record['results']):
                decision_record = record
                break
        
        if not decision_record:
            raise ValueError(f"未找到决策记录: {decision_id}")
        
        # 生成报告
        result = next(r for r in decision_record['results'] if r.option_id == decision_id)
        
        report = {
            'decision_id': decision_id,
            'timestamp': decision_record['timestamp'].isoformat(),
            'context': decision_record['context'],
            'result': {
                'option_id': result.option_id,
                'score': result.score,
                'confidence': result.confidence,
                'risk_level': result.risk_level,
                'uncertainty': result.uncertainty,
                'explanation': result.explanation,
                'metadata': result.metadata
            },
            'criteria': [{'name': c.name, 'weight': c.weight, 'type': c.type} for c in self.criteria],
            'all_options': [{'id': o.id, 'name': o.name, 'criteria_values': o.criteria_values} for o in self.options]
        }
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"决策报告已导出到: {output_file}")
    
    def reset(self):
        """重置引擎状态"""
        self.criteria.clear()
        self.options.clear()
        self.decision_history.clear()
        self.learning_engine = LearningEngine()
        logger.info("引擎状态已重置")

# 使用示例和测试函数
def example_usage():
    """使用示例"""
    # 创建决策引擎
    engine = IntelligentDecisionEngine()
    
    # 定义决策标准
    criteria = [
        DecisionCriteria("成本", 0.3, "cost", "实施成本", 0, 100),
        DecisionCriteria("效益", 0.4, "benefit", "预期效益", 0, 100),
        DecisionCriteria("风险", 0.2, "cost", "风险水平", 0, 1),
        DecisionCriteria("时间", 0.1, "cost", "实施时间", 1, 12)
    ]
    
    # 定义决策选项
    options = [
        DecisionOption("A", "方案A", "传统方案", 
                      {"成本": 60, "效益": 70, "风险": 0.3, "时间": 6},
                      {"technical_complexity": 0.3, "technology_maturity": 0.8}),
        DecisionOption("B", "方案B", "创新方案",
                      {"成本": 80, "效益": 90, "风险": 0.6, "时间": 8},
                      {"technical_complexity": 0.7, "technology_maturity": 0.4}),
        DecisionOption("C", "方案C", "平衡方案",
                      {"成本": 50, "效益": 60, "risk": 0.2, "time": 4},
                      {"technical_complexity": 0.5, "technology_maturity": 0.6})
    ]
    
    # 添加标准和选项
    engine.add_criteria(criteria)
    engine.add_options(options)
    
    # 执行决策
    context = {
        "market_volatility": 0.4,
        "competition_intensity": 0.6
    }
    
    result = engine.make_decision(context)
    
    print(f"最佳决策: {result.option_id}")
    print(f"综合评分: {result.score:.3f}")
    print(f"置信度: {result.confidence:.3f}")
    print(f"风险水平: {result.risk_level:.3f}")
    print(f"不确定性: {result.uncertainty:.3f}")
    print(f"决策解释:\n{result.explanation}")
    
    # 导出报告
    engine.export_decision_report(result.option_id, "decision_report.json")
    
    return engine, result

if __name__ == "__main__":
    # 运行示例
    engine, result = example_usage()
    
    # 模拟学习反馈
    print("\n模拟学习反馈...")
    engine.learn_from_feedback(result.option_id, 0.85, {"feedback_time": datetime.now()})
    
    # 获取学习洞察
    insights = engine.get_learning_insights()
    print(f"学习洞察: {insights}")