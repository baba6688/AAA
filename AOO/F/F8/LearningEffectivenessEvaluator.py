# -*- coding: utf-8 -*-
"""
F8学习效果评估器
Learning Effectiveness Evaluator

实现多维度学习效果评估框架，包括学习效率和效果分析、
学习进度跟踪和预测、学习质量评估和验证、学习成果评估和应用、
个性化学习建议和改进方案、学习效果综合报告生成等功能。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 内联定义经验存储相关类
class ExperienceType(Enum):
    """经验类型枚举"""
    SUCCESS = "成功经验"
    FAILURE = "失败教训"
    OPTIMIZATION = "优化路径"
    INNOVATION = "创新发现"
    ADAPTATION = "适应经验"
    EVOLUTION = "进化经验"

class LearningLevel(Enum):
    """学习层次枚举"""
    PARAMETER = 1  # 参数层
    STRATEGY = 2   # 策略层
    META = 3       # 元学习层
    COGNITIVE = 4  # 认知层
    CONSCIOUSNESS = 5  # 意识层

@dataclass
class ExperienceRecord:
    """经验记录数据结构"""
    id: str
    timestamp: datetime
    experience_type: ExperienceType
    learning_level: LearningLevel
    content: Dict[str, Any]
    context: Dict[str, Any]
    outcome: Dict[str, Any]
    confidence: float
    relevance_score: float
    tags: List[str] = field(default_factory=list)
    parent_experiences: List[str] = field(default_factory=list)
    child_experiences: List[str] = field(default_factory=list)

class ExperienceStorage:
    """简化的经验存储管理器"""
    
    def __init__(self):
        self.experience_db = {}
        
    def store_experience(self, experience: ExperienceRecord) -> str:
        """存储经验记录"""
        self.experience_db[experience.id] = experience
        return experience.id
    
    def retrieve_relevant_experiences(self, context: Dict[str, Any], 
                                    experience_type: Optional[ExperienceType] = None,
                                    learning_level: Optional[LearningLevel] = None,
                                    limit: int = 10) -> List[ExperienceRecord]:
        """检索相关经验"""
        relevant_experiences = []
        for exp in self.experience_db.values():
            if experience_type and exp.experience_type != experience_type:
                continue
            if learning_level and exp.learning_level != learning_level:
                continue
            relevant_experiences.append(exp)
        return relevant_experiences[:limit]

class EvaluationDimension(Enum):
    """评估维度枚举"""
    EFFICIENCY = "学习效率"
    EFFECTIVENESS = "学习效果"
    QUALITY = "学习质量"
    PROGRESS = "学习进度"
    APPLICATION = "应用能力"
    RETENTION = "知识保持"
    TRANSFER = "知识迁移"
    INNOVATION = "创新应用"

class QualityLevel(Enum):
    """质量等级枚举"""
    EXCELLENT = "优秀"
    GOOD = "良好"
    AVERAGE = "中等"
    POOR = "较差"
    FAIL = "不合格"

class PredictionModel(Enum):
    """预测模型枚举"""
    LINEAR_REGRESSION = "线性回归"
    POLYNOMIAL = "多项式回归"
    EXPONENTIAL = "指数回归"
    LOGISTIC = "逻辑回归"
    NEURAL_NETWORK = "神经网络"

@dataclass
class LearningMetric:
    """学习指标数据结构"""
    name: str
    value: float
    timestamp: datetime
    dimension: EvaluationDimension
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningSession:
    """学习会话数据结构"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: float
    learning_objectives: List[str]
    activities: List[Dict[str, Any]]
    metrics: List[LearningMetric]
    outcomes: Dict[str, Any]
    context: Dict[str, Any]

@dataclass
class EffectivenessScore:
    """效果评分数据结构"""
    overall_score: float
    dimension_scores: Dict[EvaluationDimension, float]
    confidence_level: float
    improvement_suggestions: List[str]
    timestamp: datetime

@dataclass
class ProgressPrediction:
    """进度预测数据结构"""
    predicted_score: float
    confidence_interval: Tuple[float, float]
    timeline: datetime
    model_used: PredictionModel
    factors: Dict[str, float]

class LearningEffectivenessEvaluator:
    """学习效果评估器主类"""
    
    def __init__(self):
        self.evaluator_id = f"evaluator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experience_storage = ExperienceStorage()
        self.learning_sessions = {}
        self.evaluation_history = []
        self.prediction_models = {}
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 0.9,
            QualityLevel.GOOD: 0.75,
            QualityLevel.AVERAGE: 0.6,
            QualityLevel.POOR: 0.4
        }
        self._initialize_prediction_models()
    
    def _initialize_prediction_models(self):
        """初始化预测模型"""
        # 为每个评估维度创建基础预测模型
        for dimension in EvaluationDimension:
            self.prediction_models[dimension] = {
                'model_type': PredictionModel.LINEAR_REGRESSION,
                'parameters': {},
                'accuracy': 0.0,
                'last_updated': datetime.now()
            }
    
    def evaluate_learning_session(self, session: LearningSession) -> EffectivenessScore:
        """评估学习会话效果"""
        print(f"开始评估学习会话: {session.session_id}")
        
        # 1. 多维度效果评估
        dimension_scores = self._evaluate_dimensions(session)
        
        # 2. 综合评分计算
        overall_score = self._calculate_overall_score(dimension_scores)
        
        # 3. 质量等级判断
        quality_level = self._determine_quality_level(overall_score)
        
        # 4. 改进建议生成
        suggestions = self._generate_improvement_suggestions(session, dimension_scores, quality_level)
        
        # 5. 创建效果评分
        effectiveness_score = EffectivenessScore(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            confidence_level=self._calculate_confidence_level(session),
            improvement_suggestions=suggestions,
            timestamp=datetime.now()
        )
        
        # 6. 保存评估结果
        self.evaluation_history.append(effectiveness_score)
        
        print(f"评估完成，综合得分: {overall_score:.2f}")
        return effectiveness_score
    
    def _evaluate_dimensions(self, session: LearningSession) -> Dict[EvaluationDimension, float]:
        """评估各个维度"""
        dimension_scores = {}
        
        # 学习效率评估
        dimension_scores[EvaluationDimension.EFFICIENCY] = self._evaluate_efficiency(session)
        
        # 学习效果评估
        dimension_scores[EvaluationDimension.EFFECTIVENESS] = self._evaluate_effectiveness(session)
        
        # 学习质量评估
        dimension_scores[EvaluationDimension.QUALITY] = self._evaluate_quality(session)
        
        # 学习进度评估
        dimension_scores[EvaluationDimension.PROGRESS] = self._evaluate_progress(session)
        
        # 应用能力评估
        dimension_scores[EvaluationDimension.APPLICATION] = self._evaluate_application(session)
        
        # 知识保持评估
        dimension_scores[EvaluationDimension.RETENTION] = self._evaluate_retention(session)
        
        # 知识迁移评估
        dimension_scores[EvaluationDimension.TRANSFER] = self._evaluate_transfer(session)
        
        # 创新应用评估
        dimension_scores[EvaluationDimension.INNOVATION] = self._evaluate_innovation(session)
        
        return dimension_scores
    
    def _evaluate_efficiency(self, session: LearningSession) -> float:
        """评估学习效率"""
        # 基于时间和完成度的效率计算
        expected_duration = len(session.learning_objectives) * 30  # 假设每个目标30分钟
        actual_duration = session.duration
        
        if actual_duration <= 0:
            return 0.0
        
        time_efficiency = min(expected_duration / actual_duration, 1.0)
        
        # 基于活动完成度的效率
        completed_activities = sum(1 for activity in session.activities 
                                 if activity.get('completed', False))
        activity_efficiency = completed_activities / max(len(session.activities), 1)
        
        # 综合效率
        efficiency_score = (time_efficiency * 0.6 + activity_efficiency * 0.4)
        
        return min(efficiency_score, 1.0)
    
    def _evaluate_effectiveness(self, session: LearningSession) -> float:
        """评估学习效果"""
        # 基于学习目标达成情况
        objective_scores = []
        for objective in session.learning_objectives:
            score = session.outcomes.get(f"objective_{objective}", 0.5)
            objective_scores.append(score)
        
        objective_effectiveness = np.mean(objective_scores) if objective_scores else 0.0
        
        # 基于测试或评估结果
        test_scores = session.outcomes.get('test_scores', [])
        test_effectiveness = np.mean(test_scores) if test_scores else 0.0
        
        # 基于实践应用效果
        application_scores = session.outcomes.get('application_scores', [])
        application_effectiveness = np.mean(application_scores) if application_scores else 0.0
        
        # 综合效果评分
        effectiveness_score = (
            objective_effectiveness * 0.4 +
            test_effectiveness * 0.4 +
            application_effectiveness * 0.2
        )
        
        return min(effectiveness_score, 1.0)
    
    def _evaluate_quality(self, session: LearningSession) -> float:
        """评估学习质量"""
        quality_indicators = []
        
        # 1. 学习深度指标
        depth_score = self._calculate_learning_depth(session)
        quality_indicators.append(depth_score)
        
        # 2. 知识整合度
        integration_score = self._calculate_knowledge_integration(session)
        quality_indicators.append(integration_score)
        
        # 3. 理解准确性
        accuracy_score = self._calculate_understanding_accuracy(session)
        quality_indicators.append(accuracy_score)
        
        # 4. 批判性思维
        critical_thinking_score = self._calculate_critical_thinking(session)
        quality_indicators.append(critical_thinking_score)
        
        return np.mean(quality_indicators)
    
    def _evaluate_progress(self, session: LearningSession) -> float:
        """评估学习进度"""
        # 获取历史进度数据
        historical_progress = self._get_historical_progress()
        
        if not historical_progress:
            return 0.5  # 基础分数
        
        # 计算进度趋势
        current_metrics = [m.value for m in session.metrics 
                          if m.dimension == EvaluationDimension.PROGRESS]
        
        if not current_metrics:
            return 0.5
        
        current_progress = np.mean(current_metrics)
        
        # 计算相对于历史的表现
        progress_improvement = self._calculate_progress_improvement(
            current_progress, historical_progress
        )
        
        return min(progress_improvement, 1.0)
    
    def _evaluate_application(self, session: LearningSession) -> float:
        """评估应用能力"""
        application_indicators = []
        
        # 1. 实际应用次数
        practical_applications = session.outcomes.get('practical_applications', 0)
        max_applications = session.outcomes.get('max_applications', 10)
        application_rate = practical_applications / max(max_applications, 1)
        application_indicators.append(application_rate)
        
        # 2. 应用成功率
        success_rate = session.outcomes.get('application_success_rate', 0.0)
        application_indicators.append(success_rate)
        
        # 3. 创新应用
        innovative_applications = session.outcomes.get('innovative_applications', 0)
        innovation_score = min(innovative_applications / 5, 1.0)  # 假设5个创新应用为满分
        application_indicators.append(innovation_score)
        
        return np.mean(application_indicators)
    
    def _evaluate_retention(self, session: LearningSession) -> float:
        """评估知识保持"""
        # 基于复习频率和间隔
        review_frequency = session.outcomes.get('review_frequency', 0)
        optimal_frequency = 3  # 假设最优复习频率为3次/周
        
        frequency_score = min(review_frequency / optimal_frequency, 1.0)
        
        # 基于记忆测试结果
        memory_test_scores = session.outcomes.get('memory_test_scores', [])
        memory_score = np.mean(memory_test_scores) if memory_test_scores else 0.5
        
        # 基于遗忘曲线分析
        forgetting_curve_score = self._analyze_forgetting_curve(session)
        
        retention_score = (
            frequency_score * 0.3 +
            memory_score * 0.5 +
            forgetting_curve_score * 0.2
        )
        
        return min(retention_score, 1.0)
    
    def _evaluate_transfer(self, session: LearningSession) -> float:
        """评估知识迁移"""
        transfer_indicators = []
        
        # 1. 跨领域应用
        cross_domain_applications = session.outcomes.get('cross_domain_applications', 0)
        transfer_score = min(cross_domain_applications / 3, 1.0)  # 假设3次跨领域应用为满分
        transfer_indicators.append(transfer_score)
        
        # 2. 类比推理能力
        analogy_score = session.outcomes.get('analogy_reasoning_score', 0.5)
        transfer_indicators.append(analogy_score)
        
        # 3. 问题解决迁移
        problem_solving_transfer = session.outcomes.get('problem_solving_transfer', 0.5)
        transfer_indicators.append(problem_solving_transfer)
        
        return np.mean(transfer_indicators)
    
    def _evaluate_innovation(self, session: LearningSession) -> float:
        """评估创新应用"""
        innovation_indicators = []
        
        # 1. 原创性想法
        original_ideas = session.outcomes.get('original_ideas', 0)
        originality_score = min(original_ideas / 5, 1.0)  # 假设5个原创想法为满分
        innovation_indicators.append(originality_score)
        
        # 2. 改进建议
        improvement_suggestions = session.outcomes.get('improvement_suggestions', 0)
        improvement_score = min(improvement_suggestions / 3, 1.0)
        innovation_indicators.append(improvement_score)
        
        # 3. 创新应用案例
        innovative_cases = session.outcomes.get('innovative_cases', 0)
        case_score = min(innovative_cases / 2, 1.0)
        innovation_indicators.append(case_score)
        
        return np.mean(innovation_indicators)
    
    def _calculate_overall_score(self, dimension_scores: Dict[EvaluationDimension, float]) -> float:
        """计算综合评分"""
        # 权重配置
        weights = {
            EvaluationDimension.EFFICIENCY: 0.15,
            EvaluationDimension.EFFECTIVENESS: 0.25,
            EvaluationDimension.QUALITY: 0.20,
            EvaluationDimension.PROGRESS: 0.15,
            EvaluationDimension.APPLICATION: 0.10,
            EvaluationDimension.RETENTION: 0.08,
            EvaluationDimension.TRANSFER: 0.05,
            EvaluationDimension.INNOVATION: 0.02
        }
        
        weighted_score = sum(
            dimension_scores[dim] * weights.get(dim, 0.1) 
            for dim in dimension_scores
        )
        
        return min(weighted_score, 1.0)
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """确定质量等级"""
        for level, threshold in sorted(self.quality_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return level
        return QualityLevel.FAIL
    
    def _generate_improvement_suggestions(self, session: LearningSession, 
                                        dimension_scores: Dict[EvaluationDimension, float],
                                        quality_level: QualityLevel) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        # 基于低分维度生成建议
        low_performing_dimensions = [
            (dim, score) for dim, score in dimension_scores.items() 
            if score < 0.6
        ]
        
        for dimension, score in low_performing_dimensions:
            if dimension == EvaluationDimension.EFFICIENCY:
                suggestions.append("建议优化学习计划，合理分配时间，提高学习效率")
            elif dimension == EvaluationDimension.EFFECTIVENESS:
                suggestions.append("建议加强实践练习，检验学习效果，及时调整学习策略")
            elif dimension == EvaluationDimension.QUALITY:
                suggestions.append("建议深入理解核心概念，加强知识间的联系和整合")
            elif dimension == EvaluationDimension.PROGRESS:
                suggestions.append("建议制定阶段性目标，跟踪学习进度，及时调整学习路径")
            elif dimension == EvaluationDimension.APPLICATION:
                suggestions.append("建议增加实际应用机会，将理论知识转化为实践技能")
            elif dimension == EvaluationDimension.RETENTION:
                suggestions.append("建议制定复习计划，采用间隔重复等记忆策略")
            elif dimension == EvaluationDimension.TRANSFER:
                suggestions.append("建议练习跨领域思考，培养知识迁移和应用能力")
            elif dimension == EvaluationDimension.INNOVATION:
                suggestions.append("建议鼓励原创思考，尝试改进现有方法或创造新方法")
        
        # 基于质量等级生成建议
        if quality_level in [QualityLevel.POOR, QualityLevel.FAIL]:
            suggestions.append("建议重新评估学习方法和目标，寻求专业指导")
        elif quality_level == QualityLevel.AVERAGE:
            suggestions.append("建议加强薄弱环节的训练，向优秀学习者学习经验")
        
        return suggestions
    
    def _calculate_confidence_level(self, session: LearningSession) -> float:
        """计算评估置信度"""
        confidence_factors = []
        
        # 基于数据完整性
        data_completeness = len(session.metrics) / 10  # 假设10个指标为满分
        confidence_factors.append(min(data_completeness, 1.0))
        
        # 基于会话时长
        if session.duration > 0:
            duration_factor = min(session.duration / 120, 1.0)  # 2小时为满分
            confidence_factors.append(duration_factor)
        
        # 基于活动数量
        activity_factor = min(len(session.activities) / 5, 1.0)  # 5个活动为满分
        confidence_factors.append(activity_factor)
        
        return np.mean(confidence_factors)
    
    def predict_learning_progress(self, student_id: str, 
                                target_date: datetime) -> ProgressPrediction:
        """预测学习进度"""
        print(f"开始预测学习进度，目标日期: {target_date}")
        
        # 获取历史数据
        historical_data = self._get_student_historical_data(student_id)
        
        if len(historical_data) < 3:
            # 数据不足，返回基础预测
            return ProgressPrediction(
                predicted_score=0.5,
                confidence_interval=(0.3, 0.7),
                timeline=target_date,
                model_used=PredictionModel.LINEAR_REGRESSION,
                factors={'data_insufficient': 1.0}
            )
        
        # 选择合适的预测模型
        model_type = self._select_prediction_model(historical_data)
        
        # 执行预测
        prediction_result = self._execute_prediction(
            historical_data, target_date, model_type
        )
        
        print(f"预测完成，预测得分: {prediction_result.predicted_score:.2f}")
        return prediction_result
    
    def _get_student_historical_data(self, student_id: str) -> List[Dict[str, Any]]:
        """获取学生历史数据"""
        # 从经验存储中获取相关数据
        relevant_experiences = self.experience_storage.retrieve_relevant_experiences(
            context={'student_id': student_id},
            limit=50
        )
        
        historical_data = []
        for exp in relevant_experiences:
            data_point = {
                'timestamp': exp.timestamp,
                'score': exp.outcome.get('learning_score', 0.5),
                'duration': exp.content.get('duration', 0),
                'activities': len(exp.content.get('activities', [])),
                'dimension_scores': exp.outcome.get('dimension_scores', {})
            }
            historical_data.append(data_point)
        
        # 按时间排序
        historical_data.sort(key=lambda x: x['timestamp'])
        return historical_data
    
    def _select_prediction_model(self, historical_data: List[Dict[str, Any]]) -> PredictionModel:
        """选择预测模型"""
        if len(historical_data) < 5:
            return PredictionModel.LINEAR_REGRESSION
        
        # 简单的模型选择逻辑
        scores = [d['score'] for d in historical_data]
        score_variance = np.var(scores)
        
        if score_variance > 0.1:
            return PredictionModel.POLYNOMIAL
        else:
            return PredictionModel.LINEAR_REGRESSION
    
    def _execute_prediction(self, historical_data: List[Dict[str, Any]], 
                          target_date: datetime, model_type: PredictionModel) -> ProgressPrediction:
        """执行预测"""
        # 准备数据
        dates = [(d['timestamp'] - historical_data[0]['timestamp']).days 
                for d in historical_data]
        scores = [d['score'] for d in historical_data]
        
        if model_type == PredictionModel.LINEAR_REGRESSION:
            # 线性回归预测
            if len(scores) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(dates, scores)
                
                target_days = (target_date - historical_data[0]['timestamp']).days
                predicted_score = slope * target_days + intercept
                predicted_score = max(0.0, min(1.0, predicted_score))  # 限制在[0,1]范围内
                
                # 计算置信区间
                confidence_interval = (
                    max(0.0, predicted_score - 2 * std_err),
                    min(1.0, predicted_score + 2 * std_err)
                )
                
                return ProgressPrediction(
                    predicted_score=predicted_score,
                    confidence_interval=confidence_interval,
                    timeline=target_date,
                    model_used=model_type,
                    factors={'r_squared': r_value**2, 'slope': slope}
                )
        
        # 默认返回中等分数
        return ProgressPrediction(
            predicted_score=0.5,
            confidence_interval=(0.3, 0.7),
            timeline=target_date,
            model_used=model_type,
            factors={'default': 1.0}
        )
    
    def generate_learning_report(self, student_id: str, 
                               start_date: datetime, 
                               end_date: datetime) -> Dict[str, Any]:
        """生成学习效果报告"""
        print(f"生成学习报告: {student_id}, {start_date.date()} - {end_date.date()}")
        
        # 获取相关数据
        sessions = self._get_student_sessions(student_id, start_date, end_date)
        evaluations = self._get_student_evaluations(student_id, start_date, end_date)
        
        if not sessions:
            return {'error': '指定时间段内没有学习数据'}
        
        # 生成报告内容
        report = {
            'student_id': student_id,
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': self._generate_summary_statistics(sessions, evaluations),
            'dimension_analysis': self._analyze_dimensions(evaluations),
            'progress_trend': self._analyze_progress_trend(evaluations),
            'quality_assessment': self._assess_learning_quality(evaluations),
            'achievement_highlights': self._identify_achievements(evaluations),
            'improvement_areas': self._identify_improvement_areas(evaluations),
            'recommendations': self._generate_recommendations(evaluations),
            'predictions': self._generate_predictions(student_id, end_date),
            'generated_at': datetime.now().isoformat()
        }
        
        print("学习报告生成完成")
        return report
    
    def _get_student_sessions(self, student_id: str, start_date: datetime, 
                            end_date: datetime) -> List[LearningSession]:
        """获取学生学习会话"""
        # 模拟数据获取逻辑
        relevant_sessions = []
        for session in self.learning_sessions.values():
            if (session.context.get('student_id') == student_id and
                start_date <= session.start_time <= end_date):
                relevant_sessions.append(session)
        
        return relevant_sessions
    
    def _get_student_evaluations(self, student_id: str, start_date: datetime,
                               end_date: datetime) -> List[EffectivenessScore]:
        """获取学生评估结果"""
        relevant_evaluations = []
        for evaluation in self.evaluation_history:
            if start_date <= evaluation.timestamp <= end_date:
                relevant_evaluations.append(evaluation)
        
        return relevant_evaluations
    
    def _generate_summary_statistics(self, sessions: List[LearningSession],
                                   evaluations: List[EffectivenessScore]) -> Dict[str, Any]:
        """生成汇总统计"""
        if not evaluations:
            return {'total_sessions': len(sessions), 'average_score': 0.0}
        
        scores = [eval.overall_score for eval in evaluations]
        
        return {
            'total_sessions': len(sessions),
            'total_evaluations': len(evaluations),
            'average_score': np.mean(scores),
            'score_variance': np.var(scores),
            'highest_score': max(scores),
            'lowest_score': min(scores),
            'improvement_trend': self._calculate_improvement_trend(scores)
        }
    
    def _analyze_dimensions(self, evaluations: List[EffectivenessScore]) -> Dict[str, Any]:
        """分析各维度表现"""
        if not evaluations:
            return {}
        
        dimension_analysis = {}
        
        for dimension in EvaluationDimension:
            scores = [eval.dimension_scores.get(dimension, 0.0) for eval in evaluations]
            if scores:
                dimension_analysis[dimension.value] = {
                    'average_score': np.mean(scores),
                    'score_trend': self._calculate_improvement_trend(scores),
                    'consistency': 1.0 - np.var(scores)  # 一致性指标
                }
        
        return dimension_analysis
    
    def _analyze_progress_trend(self, evaluations: List[EffectivenessScore]) -> Dict[str, Any]:
        """分析进度趋势"""
        if len(evaluations) < 2:
            return {'trend': 'insufficient_data'}
        
        scores = [eval.overall_score for eval in evaluations]
        trend = self._calculate_improvement_trend(scores)
        
        return {
            'trend': trend,
            'slope': self._calculate_trend_slope(scores),
            'confidence': self._calculate_trend_confidence(scores)
        }
    
    def _assess_learning_quality(self, evaluations: List[EffectivenessScore]) -> Dict[str, Any]:
        """评估学习质量"""
        if not evaluations:
            return {'quality_level': 'unknown'}
        
        recent_scores = [eval.overall_score for eval in evaluations[-5:]]  # 最近5次
        avg_recent_score = np.mean(recent_scores)
        
        quality_level = self._determine_quality_level(avg_recent_score)
        
        return {
            'quality_level': quality_level.value,
            'average_score': avg_recent_score,
            'consistency': 1.0 - np.var(recent_scores),
            'stability': self._calculate_stability(recent_scores)
        }
    
    def _identify_achievements(self, evaluations: List[EffectivenessScore]) -> List[str]:
        """识别学习成就"""
        achievements = []
        
        if not evaluations:
            return achievements
        
        scores = [eval.overall_score for eval in evaluations]
        max_score = max(scores)
        
        # 识别高分成就
        if max_score >= 0.9:
            achievements.append("达到优秀学习水平")
        
        # 识别进步成就
        if len(scores) >= 3:
            recent_avg = np.mean(scores[-3:])
            early_avg = np.mean(scores[:3])
            if recent_avg - early_avg > 0.2:
                achievements.append("显著进步表现")
        
        # 识别稳定性成就
        score_variance = np.var(scores)
        if score_variance < 0.05:
            achievements.append("学习表现稳定")
        
        return achievements
    
    def _identify_improvement_areas(self, evaluations: List[EffectivenessScore]) -> List[str]:
        """识别改进领域"""
        if not evaluations:
            return []
        
        improvement_areas = []
        
        # 分析各维度表现
        dimension_averages = {}
        for dimension in EvaluationDimension:
            scores = [eval.dimension_scores.get(dimension, 0.0) for eval in evaluations]
            if scores:
                dimension_averages[dimension] = np.mean(scores)
        
        # 找出表现较差的维度
        for dimension, avg_score in dimension_averages.items():
            if avg_score < 0.6:
                improvement_areas.append(f"{dimension.value}需要加强")
        
        return improvement_areas
    
    def _generate_recommendations(self, evaluations: List[EffectivenessScore]) -> List[str]:
        """生成个性化建议"""
        recommendations = []
        
        if not evaluations:
            return ["建议开始系统化学习，建立学习记录"]
        
        # 基于总体表现生成建议
        recent_scores = [eval.overall_score for eval in evaluations[-3:]]
        avg_recent_score = np.mean(recent_scores)
        
        if avg_recent_score < 0.5:
            recommendations.append("建议重新评估学习方法，寻求专业指导")
        elif avg_recent_score < 0.7:
            recommendations.append("建议加强基础训练，巩固核心概念")
        else:
            recommendations.append("表现良好，建议挑战更高难度的学习内容")
        
        # 基于维度分析生成建议
        recommendations.extend(self._generate_dimension_specific_recommendations(evaluations))
        
        return recommendations
    
    def _generate_predictions(self, student_id: str, target_date: datetime) -> Dict[str, Any]:
        """生成预测信息"""
        predictions = {}
        
        # 短期预测（1周内）
        short_term_prediction = self.predict_learning_progress(
            student_id, target_date + timedelta(days=7)
        )
        predictions['short_term'] = {
            'predicted_score': short_term_prediction.predicted_score,
            'confidence_interval': short_term_prediction.confidence_interval
        }
        
        # 中期预测（1个月内）
        medium_term_prediction = self.predict_learning_progress(
            student_id, target_date + timedelta(days=30)
        )
        predictions['medium_term'] = {
            'predicted_score': medium_term_prediction.predicted_score,
            'confidence_interval': medium_term_prediction.confidence_interval
        }
        
        return predictions
    
    # 辅助方法
    def _calculate_learning_depth(self, session: LearningSession) -> float:
        """计算学习深度"""
        depth_indicators = []
        
        # 基于问题复杂度
        complex_questions = sum(1 for activity in session.activities 
                              if activity.get('complexity', 1) > 3)
        depth_indicators.append(min(complex_questions / 5, 1.0))
        
        # 基于反思深度
        reflection_depth = session.outcomes.get('reflection_depth', 0.5)
        depth_indicators.append(reflection_depth)
        
        return np.mean(depth_indicators)
    
    def _calculate_knowledge_integration(self, session: LearningSession) -> float:
        """计算知识整合度"""
        integration_score = session.outcomes.get('knowledge_integration_score', 0.5)
        return integration_score
    
    def _calculate_understanding_accuracy(self, session: LearningSession) -> float:
        """计算理解准确性"""
        accuracy_score = session.outcomes.get('understanding_accuracy', 0.5)
        return accuracy_score
    
    def _calculate_critical_thinking(self, session: LearningSession) -> float:
        """计算批判性思维"""
        critical_thinking_score = session.outcomes.get('critical_thinking_score', 0.5)
        return critical_thinking_score
    
    def _get_historical_progress(self) -> List[float]:
        """获取历史进度数据"""
        progress_scores = []
        for evaluation in self.evaluation_history:
            progress_score = evaluation.dimension_scores.get(EvaluationDimension.PROGRESS, 0.0)
            progress_scores.append(progress_score)
        return progress_scores
    
    def _calculate_progress_improvement(self, current_progress: float, 
                                      historical_progress: List[float]) -> float:
        """计算进度改进"""
        if not historical_progress:
            return current_progress
        
        # 计算相对于历史平均的改进
        historical_avg = np.mean(historical_progress)
        improvement = (current_progress - historical_avg) / max(historical_avg, 0.1)
        
        return max(0.0, min(1.0, 0.5 + improvement))
    
    def _analyze_forgetting_curve(self, session: LearningSession) -> float:
        """分析遗忘曲线"""
        forgetting_rate = session.outcomes.get('forgetting_rate', 0.5)
        return 1.0 - forgetting_rate  # 遗忘率越低，保持率越高
    
    def _calculate_improvement_trend(self, scores: List[float]) -> str:
        """计算改进趋势"""
        if len(scores) < 2:
            return 'insufficient_data'
        
        # 简单线性趋势分析
        x = list(range(len(scores)))
        slope, _, r_value, _, _ = stats.linregress(x, scores)
        
        if r_value**2 < 0.3:  # 相关性太低
            return 'unstable'
        elif slope > 0.05:
            return 'improving'
        elif slope < -0.05:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_trend_slope(self, scores: List[float]) -> float:
        """计算趋势斜率"""
        if len(scores) < 2:
            return 0.0
        
        x = list(range(len(scores)))
        slope, _, _, _, _ = stats.linregress(x, scores)
        return slope
    
    def _calculate_trend_confidence(self, scores: List[float]) -> float:
        """计算趋势置信度"""
        if len(scores) < 2:
            return 0.0
        
        x = list(range(len(scores)))
        _, _, r_value, _, _ = stats.linregress(x, scores)
        return r_value**2
    
    def _calculate_stability(self, scores: List[float]) -> float:
        """计算稳定性"""
        if len(scores) < 2:
            return 0.0
        
        # 稳定性 = 1 - 变异系数
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if mean_score == 0:
            return 0.0
        
        coefficient_of_variation = std_score / mean_score
        stability = max(0.0, 1.0 - coefficient_of_variation)
        
        return stability
    
    def _generate_dimension_specific_recommendations(self, evaluations: List[EffectivenessScore]) -> List[str]:
        """生成维度特定建议"""
        recommendations = []
        
        # 计算各维度平均表现
        dimension_averages = {}
        for dimension in EvaluationDimension:
            scores = [eval.dimension_scores.get(dimension, 0.0) for eval in evaluations]
            if scores:
                dimension_averages[dimension] = np.mean(scores)
        
        # 为低分维度生成具体建议
        for dimension, avg_score in dimension_averages.items():
            if avg_score < 0.6:
                if dimension == EvaluationDimension.EFFICIENCY:
                    recommendations.append("建议优化时间管理，使用番茄工作法提高专注度")
                elif dimension == EvaluationDimension.EFFECTIVENESS:
                    recommendations.append("建议增加实践练习，通过应用巩固理论知识")
                elif dimension == EvaluationDimension.QUALITY:
                    recommendations.append("建议深入理解概念本质，加强知识间的逻辑联系")
        
        return recommendations

# 全局学习效果评估器实例
learning_evaluator = LearningEffectivenessEvaluator()

# 使用示例和测试函数
def example_usage():
    """学习效果评估器使用示例"""
    
    # 创建示例学习会话
    session = LearningSession(
        session_id="session_001",
        start_time=datetime.now() - timedelta(hours=2),
        end_time=datetime.now(),
        duration=120.0,  # 2小时
        learning_objectives=["掌握Python基础", "理解面向对象编程", "完成实际项目"],
        activities=[
            {"name": "阅读教程", "completed": True, "duration": 30},
            {"name": "编写代码", "completed": True, "duration": 45},
            {"name": "项目实践", "completed": True, "duration": 45}
        ],
        metrics=[
            LearningMetric("completion_rate", 0.9, datetime.now(), 
                         EvaluationDimension.EFFICIENCY, 0.85),
            LearningMetric("test_score", 0.85, datetime.now(), 
                         EvaluationDimension.EFFECTIVENESS, 0.90)
        ],
        outcomes={
            "objective_掌握Python基础": 0.9,
            "objective_理解面向对象编程": 0.8,
            "objective_完成实际项目": 0.75,
            "test_scores": [0.85, 0.90, 0.80],
            "application_scores": [0.8, 0.85],
            "practical_applications": 3,
            "max_applications": 5,
            "application_success_rate": 0.8,
            "memory_test_scores": [0.85, 0.90],
            "cross_domain_applications": 1,
            "original_ideas": 2,
            "improvement_suggestions": 1,
            "reflection_depth": 0.8,
            "knowledge_integration_score": 0.75,
            "understanding_accuracy": 0.85,
            "critical_thinking_score": 0.7
        },
        context={
            "student_id": "student_001",
            "subject": "Python编程",
            "difficulty": "中等",
            "learning_style": "实践导向"
        }
    )
    
    # 评估学习效果
    print("=== 学习效果评估 ===")
    effectiveness_score = learning_evaluator.evaluate_learning_session(session)
    
    print(f"综合评分: {effectiveness_score.overall_score:.2f}")
    print("各维度评分:")
    for dimension, score in effectiveness_score.dimension_scores.items():
        print(f"  {dimension.value}: {score:.2f}")
    
    print(f"置信度: {effectiveness_score.confidence_level:.2f}")
    print("改进建议:")
    for suggestion in effectiveness_score.improvement_suggestions:
        print(f"  - {suggestion}")
    
    # 预测学习进度
    print("\n=== 学习进度预测 ===")
    prediction = learning_evaluator.predict_learning_progress(
        "student_001", 
        datetime.now() + timedelta(days=30)
    )
    
    print(f"预测得分: {prediction.predicted_score:.2f}")
    print(f"置信区间: {prediction.confidence_interval}")
    print(f"预测模型: {prediction.model_used.value}")
    
    # 生成学习报告
    print("\n=== 学习报告生成 ===")
    
    # 添加学习会话到评估器
    learning_evaluator.learning_sessions[session.session_id] = session
    
    report = learning_evaluator.generate_learning_report(
        "student_001",
        datetime.now() - timedelta(days=30),
        datetime.now()
    )
    
    if 'error' in report:
        print(f"报告生成错误: {report['error']}")
    else:
        print("报告摘要:")
        print(f"  平均得分: {report['summary']['average_score']:.2f}")
        print(f"  改进趋势: {report['summary']['improvement_trend']}")
        print("成就亮点:")
        for achievement in report['achievement_highlights']:
            print(f"  - {achievement}")
        print("改进领域:")
        for area in report['improvement_areas']:
            print(f"  - {area}")

if __name__ == "__main__":
    example_usage()