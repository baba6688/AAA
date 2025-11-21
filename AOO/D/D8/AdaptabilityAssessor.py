#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D8 适应能力评估器
实现多维度适应能力评估、环境适应性测试、适应速度质量量化等功能
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptabilityType(Enum):
    """适应类型枚举"""
    COGNITIVE = "认知适应"  # 思维模式适应
    BEHAVIORAL = "行为适应"  # 行为模式适应
    EMOTIONAL = "情感适应"  # 情感调节适应
    SOCIAL = "社会适应"  # 社会关系适应
    TECHNICAL = "技术适应"  # 技术技能适应
    STRATEGIC = "战略适应"  # 战略规划适应


class EnvironmentType(Enum):
    """环境类型枚举"""
    STABLE = "稳定环境"  # 变化缓慢的环境
    DYNAMIC = "动态环境"  # 持续变化的环境
    TURBULENT = "混乱环境"  # 高度不确定的环境
    PREDICTABLE = "可预测环境"  # 变化可预测的环境
    UNPREDICTABLE = "不可预测环境"  # 变化不可预测的环境


class AdaptationStrategy(Enum):
    """适应策略枚举"""
    PROACTIVE = "主动适应"  # 提前准备适应
    REACTIVE = "被动适应"  # 被动响应适应
    ANTICIPATORY = "预期适应"  # 基于预测的适应
    LEARNING = "学习适应"  # 通过学习适应
    INNOVATIVE = "创新适应"  # 通过创新适应


@dataclass
class AdaptabilityMetrics:
    """适应能力指标"""
    cognitive_flexibility: float = 0.0  # 认知灵活性
    behavioral_adaptability: float = 0.0  # 行为适应性
    emotional_regulation: float = 0.0  # 情感调节能力
    social_adaptation: float = 0.0  # 社会适应能力
    technical_learning: float = 0.0  # 技术学习能力
    strategic_thinking: float = 0.0  # 战略思维能力
    overall_score: float = 0.0  # 总体评分


@dataclass
class EnvironmentChange:
    """环境变化记录"""
    change_id: str
    change_type: str
    severity: float  # 变化严重程度 (0-1)
    frequency: float  # 变化频率 (0-1)
    predictability: float  # 可预测性 (0-1)
    impact_scope: float  # 影响范围 (0-1)
    timestamp: datetime
    description: str


@dataclass
class AdaptationPerformance:
    """适应性能记录"""
    adaptation_id: str
    environment_change_id: str
    strategy_used: AdaptationStrategy
    adaptation_time: float  # 适应时间（小时）
    adaptation_quality: float  # 适应质量 (0-1)
    success_rate: float  # 成功率 (0-1)
    resource_cost: float  # 资源成本
    outcome_score: float  # 结果评分 (0-1)
    timestamp: datetime


@dataclass
class AdaptationSuggestion:
    """适应建议"""
    suggestion_id: str
    category: str
    priority: str  # 高、中、低
    description: str
    implementation_steps: List[str]
    expected_improvement: float
    time_estimate: str
    resource_requirements: List[str]


class AdaptabilityModel:
    """适应能力模型"""
    
    def __init__(self):
        self.dimensions = {
            AdaptabilityType.COGNITIVE: {
                'weight': 0.2,
                'indicators': ['思维灵活性', '学习能力', '问题解决能力', '创新思维']
            },
            AdaptabilityType.BEHAVIORAL: {
                'weight': 0.18,
                'indicators': ['行为灵活性', '执行能力', '习惯调整', '技能转换']
            },
            AdaptabilityType.EMOTIONAL: {
                'weight': 0.15,
                'indicators': ['情绪稳定性', '压力管理', '挫折承受', '积极心态']
            },
            AdaptabilityType.SOCIAL: {
                'weight': 0.17,
                'indicators': ['沟通能力', '协作能力', '关系维护', '团队融入']
            },
            AdaptabilityType.TECHNICAL: {
                'weight': 0.15,
                'indicators': ['技术学习', '工具掌握', '系统操作', '数字素养']
            },
            AdaptabilityType.STRATEGIC: {
                'weight': 0.15,
                'indicators': ['战略思维', '规划能力', '决策质量', '风险评估']
            }
        }
        
        self.baseline_metrics = AdaptabilityMetrics()
        self.current_metrics = AdaptabilityMetrics()
        self.historical_data = []
        
    def calculate_overall_score(self, metrics: AdaptabilityMetrics) -> float:
        """计算总体适应能力评分"""
        weights = self.dimensions
        score = (
            metrics.cognitive_flexibility * weights[AdaptabilityType.COGNITIVE]['weight'] +
            metrics.behavioral_adaptability * weights[AdaptabilityType.BEHAVIORAL]['weight'] +
            metrics.emotional_regulation * weights[AdaptabilityType.EMOTIONAL]['weight'] +
            metrics.social_adaptation * weights[AdaptabilityType.SOCIAL]['weight'] +
            metrics.technical_learning * weights[AdaptabilityType.TECHNICAL]['weight'] +
            metrics.strategic_thinking * weights[AdaptabilityType.STRATEGIC]['weight']
        )
        return min(max(score, 0.0), 1.0)
    
    def update_metrics(self, new_metrics: AdaptabilityMetrics):
        """更新适应能力指标"""
        self.baseline_metrics = self.current_metrics
        self.current_metrics = new_metrics
        self.current_metrics.overall_score = self.calculate_overall_score(new_metrics)
        
        # 保存历史数据
        self.historical_data.append({
            'timestamp': datetime.now(),
            'metrics': asdict(new_metrics)
        })


class AdaptabilityAssessor:
    """适应能力评估器主类"""
    
    def __init__(self):
        self.model = AdaptabilityModel()
        self.environment_changes = []
        self.adaptation_performances = []
        self.suggestions = []
        self.alerts = []
        self.tracking_data = []
        
        # 评估参数
        self.adaptation_threshold = 0.7  # 适应能力阈值
        self.performance_threshold = 0.6  # 性能阈值
        self.warning_threshold = 0.5  # 预警阈值
        
    def build_adaptability_model(self, initial_assessment: Dict[str, float]) -> AdaptabilityMetrics:
        """构建适应能力模型
        
        Args:
            initial_assessment: 初始评估数据
            
        Returns:
            AdaptabilityMetrics: 适应能力指标
        """
        logger.info("构建适应能力模型...")
        
        # 创建初始指标
        metrics = AdaptabilityMetrics(
            cognitive_flexibility=initial_assessment.get('cognitive_flexibility', 0.5),
            behavioral_adaptability=initial_assessment.get('behavioral_adaptability', 0.5),
            emotional_regulation=initial_assessment.get('emotional_regulation', 0.5),
            social_adaptation=initial_assessment.get('social_adaptation', 0.5),
            technical_learning=initial_assessment.get('technical_learning', 0.5),
            strategic_thinking=initial_assessment.get('strategic_thinking', 0.5)
        )
        
        # 计算总体评分
        metrics.overall_score = self.model.calculate_overall_score(metrics)
        
        # 更新模型
        self.model.update_metrics(metrics)
        
        logger.info(f"适应能力模型构建完成，总体评分: {metrics.overall_score:.3f}")
        return metrics
    
    def assess_environment_adaptability(self, environment_type: EnvironmentType, 
                                      change_frequency: float,
                                      change_magnitude: float) -> Dict[str, Any]:
        """评估环境变化适应性
        
        Args:
            environment_type: 环境类型
            change_frequency: 变化频率
            change_magnitude: 变化幅度
            
        Returns:
            Dict: 环境适应性评估结果
        """
        logger.info(f"评估环境适应性: {environment_type.value}")
        
        # 环境适应性系数
        environment_coefficients = {
            EnvironmentType.STABLE: {'base_resistance': 0.8, 'adaptation_speed': 0.6},
            EnvironmentType.DYNAMIC: {'base_resistance': 0.6, 'adaptation_speed': 0.8},
            EnvironmentType.TURBULENT: {'base_resistance': 0.3, 'adaptation_speed': 0.9},
            EnvironmentType.PREDICTABLE: {'base_resistance': 0.7, 'adaptation_speed': 0.7},
            EnvironmentType.UNPREDICTABLE: {'base_resistance': 0.4, 'adaptation_speed': 0.8}
        }
        
        coeff = environment_coefficients[environment_type]
        
        # 计算适应性指标
        resistance_score = coeff['base_resistance'] * (1 - change_magnitude)
        speed_score = coeff['adaptation_speed'] * (1 - change_frequency * 0.5)
        overall_adaptability = (resistance_score + speed_score) / 2
        
        # 记录环境变化
        change = EnvironmentChange(
            change_id=f"env_{len(self.environment_changes)}",
            change_type=environment_type.value,
            severity=change_magnitude,
            frequency=change_frequency,
            predictability=0.5 if environment_type == EnvironmentType.UNPREDICTABLE else 0.8,
            impact_scope=1.0,
            timestamp=datetime.now(),
            description=f"{environment_type.value}环境变化"
        )
        self.environment_changes.append(change)
        
        result = {
            'environment_type': environment_type.value,
            'resistance_score': resistance_score,
            'adaptation_speed_score': speed_score,
            'overall_adaptability': overall_adaptability,
            'recommendation': self._get_adaptation_recommendation(overall_adaptability),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"环境适应性评估完成: {overall_adaptability:.3f}")
        return result
    
    def assess_adaptation_speed_quality(self, adaptation_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估适应速度和质量
        
        Args:
            adaptation_data: 适应数据
            
        Returns:
            Dict: 适应速度和质量评估结果
        """
        logger.info("评估适应速度和适应质量...")
        
        # 适应速度评估
        adaptation_time = adaptation_data.get('adaptation_time', 24)  # 小时
        expected_time = adaptation_data.get('expected_time', 24)
        
        speed_score = max(0, 1 - (adaptation_time - expected_time) / expected_time)
        speed_score = min(speed_score, 1.0)
        
        # 适应质量评估
        quality_indicators = adaptation_data.get('quality_indicators', {})
        quality_weights = {
            'accuracy': 0.3,  # 准确性
            'completeness': 0.25,  # 完整性
            'efficiency': 0.25,  # 效率
            'sustainability': 0.2  # 持续性
        }
        
        quality_score = sum(
            quality_indicators.get(indicator, 0.5) * weight
            for indicator, weight in quality_weights.items()
        )
        
        # 综合适应性能
        overall_performance = (speed_score + quality_score) / 2
        
        # 记录适应性能
        performance = AdaptationPerformance(
            adaptation_id=f"adapt_{len(self.adaptation_performances)}",
            environment_change_id=adaptation_data.get('environment_change_id', ''),
            strategy_used=adaptation_data.get('strategy', AdaptationStrategy.REACTIVE),
            adaptation_time=adaptation_time,
            adaptation_quality=quality_score,
            success_rate=adaptation_data.get('success_rate', 0.8),
            resource_cost=adaptation_data.get('resource_cost', 1.0),
            outcome_score=overall_performance,
            timestamp=datetime.now()
        )
        self.adaptation_performances.append(performance)
        
        result = {
            'speed_score': speed_score,
            'quality_score': quality_score,
            'overall_performance': overall_performance,
            'adaptation_time': adaptation_time,
            'expected_time': expected_time,
            'speed_rating': self._rate_performance(speed_score),
            'quality_rating': self._rate_performance(quality_score),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"适应性能评估完成: {overall_performance:.3f}")
        return result
    
    def evaluate_strategy_effectiveness(self, strategy: AdaptationStrategy,
                                      performance_data: List[Dict]) -> Dict[str, Any]:
        """评估适应策略效果
        
        Args:
            strategy: 适应策略
            performance_data: 性能数据列表
            
        Returns:
            Dict: 策略效果评估结果
        """
        logger.info(f"评估策略效果: {strategy.value}")
        
        if not performance_data:
            return {'error': '没有性能数据可供评估'}
        
        # 计算策略效果指标
        success_rates = [data.get('success_rate', 0) for data in performance_data]
        efficiency_scores = [data.get('efficiency_score', 0) for data in performance_data]
        resource_costs = [data.get('resource_cost', 1) for data in performance_data]
        
        avg_success_rate = np.mean(success_rates)
        avg_efficiency = np.mean(efficiency_scores)
        avg_resource_cost = np.mean(resource_costs)
        
        # 策略效果评分
        effectiveness_score = (
            avg_success_rate * 0.4 +
            avg_efficiency * 0.35 +
            (1 - avg_resource_cost) * 0.25
        )
        
        # 策略适用性分析
        applicability_analysis = self._analyze_strategy_applicability(strategy)
        
        result = {
            'strategy': strategy.value,
            'effectiveness_score': effectiveness_score,
            'average_success_rate': avg_success_rate,
            'average_efficiency': avg_efficiency,
            'average_resource_cost': avg_resource_cost,
            'sample_size': len(performance_data),
            'applicability': applicability_analysis,
            'recommendations': self._get_strategy_recommendations(strategy, effectiveness_score),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"策略效果评估完成: {effectiveness_score:.3f}")
        return result
    
    def generate_improvement_suggestions(self, current_metrics: AdaptabilityMetrics) -> List[AdaptationSuggestion]:
        """生成适应能力提升建议
        
        Args:
            current_metrics: 当前适应能力指标
            
        Returns:
            List[AdaptationSuggestion]: 提升建议列表
        """
        logger.info("生成适应能力提升建议...")
        
        suggestions = []
        
        # 分析各维度表现
        dimension_scores = {
            '认知灵活性': current_metrics.cognitive_flexibility,
            '行为适应性': current_metrics.behavioral_adaptability,
            '情感调节': current_metrics.emotional_regulation,
            '社会适应': current_metrics.social_adaptation,
            '技术学习': current_metrics.technical_learning,
            '战略思维': current_metrics.strategic_thinking
        }
        
        # 为低分维度生成建议
        for dimension, score in dimension_scores.items():
            if score < 0.6:  # 低于60分的维度需要改进
                suggestion = self._generate_dimension_suggestion(dimension, score)
                suggestions.append(suggestion)
        
        # 生成综合提升建议
        if current_metrics.overall_score < 0.7:
            comprehensive_suggestion = self._generate_comprehensive_suggestion(current_metrics)
            suggestions.append(comprehensive_suggestion)
        
        # 保存建议
        self.suggestions.extend(suggestions)
        
        logger.info(f"生成了 {len(suggestions)} 条提升建议")
        return suggestions
    
    def setup_adaptation_alerts(self, alert_config: Dict[str, Any]) -> Dict[str, Any]:
        """设置适应能力预警机制
        
        Args:
            alert_config: 预警配置
            
        Returns:
            Dict: 预警设置结果
        """
        logger.info("设置适应能力预警机制...")
        
        # 预警阈值设置
        thresholds = {
            'low_adaptability': alert_config.get('low_adaptability_threshold', 0.5),
            'declining_performance': alert_config.get('declining_performance_threshold', -0.1),
            'high_adaptation_time': alert_config.get('high_adaptation_time_threshold', 48),
            'low_success_rate': alert_config.get('low_success_rate_threshold', 0.6)
        }
        
        # 预警规则
        alert_rules = [
            {
                'rule_id': 'adaptability_drop',
                'condition': 'overall_score < thresholds.low_adaptability',
                'message': '适应能力低于预警阈值',
                'severity': 'high'
            },
            {
                'rule_id': 'performance_decline',
                'condition': 'performance_trend < thresholds.declining_performance',
                'message': '适应性能呈下降趋势',
                'severity': 'medium'
            },
            {
                'rule_id': 'slow_adaptation',
                'condition': 'avg_adaptation_time > thresholds.high_adaptation_time',
                'message': '适应速度过慢',
                'severity': 'medium'
            },
            {
                'rule_id': 'low_success_rate',
                'condition': 'success_rate < thresholds.low_success_rate',
                'message': '适应成功率偏低',
                'severity': 'high'
            }
        ]
        
        result = {
            'thresholds': thresholds,
            'alert_rules': alert_rules,
            'alert_methods': alert_config.get('methods', ['log', 'notification']),
            'status': 'active',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("适应能力预警机制设置完成")
        return result
    
    def track_adaptation_development(self, time_period: int = 30) -> Dict[str, Any]:
        """跟踪适应能力发展
        
        Args:
            time_period: 跟踪时间周期（天）
            
        Returns:
            Dict: 适应能力发展跟踪结果
        """
        logger.info(f"跟踪适应能力发展，时间周期: {time_period}天")
        
        if not self.model.historical_data:
            return {'error': '没有历史数据可供跟踪'}
        
        # 获取时间范围内的数据
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_period)
        
        relevant_data = [
            data for data in self.model.historical_data
            if data['timestamp'] >= start_time
        ]
        
        if len(relevant_data) < 2:
            return {'error': '历史数据不足，无法进行趋势分析'}
        
        # 计算发展趋势
        trends = self._calculate_development_trends(relevant_data)
        
        # 生成发展报告
        development_report = {
            'time_period': time_period,
            'data_points': len(relevant_data),
            'overall_trend': trends['overall'],
            'dimension_trends': trends['dimensions'],
            'performance_metrics': self._calculate_performance_metrics(relevant_data),
            'milestones': self._identify_development_milestones(relevant_data),
            'recommendations': self._generate_development_recommendations(trends),
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存跟踪数据
        self.tracking_data.append(development_report)
        
        logger.info("适应能力发展跟踪完成")
        return development_report
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合评估报告
        
        Returns:
            Dict: 综合评估报告
        """
        logger.info("生成综合评估报告...")
        
        report = {
            'report_id': f"adaptability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generation_time': datetime.now().isoformat(),
            'current_metrics': asdict(self.model.current_metrics),
            'baseline_metrics': asdict(self.model.baseline_metrics),
            'improvement_progress': self._calculate_improvement_progress(),
            'environment_analysis': self._analyze_environment_patterns(),
            'strategy_performance': self._analyze_strategy_performance(),
            'recommendations': [asdict(s) for s in self.suggestions[-5:]],  # 最近5条建议
            'alert_summary': self._summarize_alerts(),
            'development_trends': self._analyze_development_trends(),
            'next_actions': self._generate_next_actions()
        }
        
        logger.info("综合评估报告生成完成")
        return report
    
    # 辅助方法
    def _get_adaptation_recommendation(self, adaptability_score: float) -> str:
        """获取适应建议"""
        if adaptability_score >= 0.8:
            return "适应能力优秀，继续保持并分享经验"
        elif adaptability_score >= 0.6:
            return "适应能力良好，可进一步提升特定维度"
        elif adaptability_score >= 0.4:
            return "适应能力中等，需要针对性训练和提升"
        else:
            return "适应能力偏低，建议寻求专业指导和系统训练"
    
    def _rate_performance(self, score: float) -> str:
        """评级性能"""
        if score >= 0.9:
            return "优秀"
        elif score >= 0.8:
            return "良好"
        elif score >= 0.7:
            return "中等"
        elif score >= 0.6:
            return "及格"
        else:
            return "不及格"
    
    def _analyze_strategy_applicability(self, strategy: AdaptationStrategy) -> Dict[str, Any]:
        """分析策略适用性"""
        applicability_map = {
            AdaptationStrategy.PROACTIVE: {
                'best_for': ['可预测环境', '稳定环境'],
                'requirements': ['前瞻性思维', '资源储备'],
                'limitations': ['资源消耗大', '预测准确性要求高']
            },
            AdaptationStrategy.REACTIVE: {
                'best_for': ['不可预测环境', '紧急情况'],
                'requirements': ['快速响应能力', '灵活性'],
                'limitations': ['效果有限', '被动应对']
            },
            AdaptationStrategy.ANTICIPATORY: {
                'best_for': ['动态环境', '复杂环境'],
                'requirements': ['数据分析能力', '预测模型'],
                'limitations': ['预测误差风险', '模型维护成本']
            },
            AdaptationStrategy.LEARNING: {
                'best_for': ['技术环境', '知识密集环境'],
                'requirements': ['学习能力', '知识管理'],
                'limitations': ['学习周期长', '知识更新快']
            },
            AdaptationStrategy.INNOVATIVE: {
                'best_for': ['创新环境', '变革环境'],
                'requirements': ['创新思维', '实验能力'],
                'limitations': ['风险较高', '资源需求大']
            }
        }
        
        return applicability_map.get(strategy, {})
    
    def _get_strategy_recommendations(self, strategy: AdaptationStrategy, effectiveness: float) -> List[str]:
        """获取策略建议"""
        if effectiveness >= 0.8:
            return [f"继续使用{strategy.value}策略，效果良好"]
        elif effectiveness >= 0.6:
            return [f"优化{strategy.value}策略实施细节"]
        else:
            return [
                f"重新评估{strategy.value}策略的适用性",
                "考虑结合其他策略使用",
                "分析失败原因并改进"
            ]
    
    def _generate_dimension_suggestion(self, dimension: str, score: float) -> AdaptationSuggestion:
        """生成维度建议"""
        suggestion_templates = {
            '认知灵活性': {
                'high_priority': [
                    '进行思维导图训练',
                    '练习多角度思考问题',
                    '学习新的思维方法'
                ],
                'medium_priority': [
                    '增加跨领域学习',
                    '参与头脑风暴活动'
                ]
            },
            '行为适应性': {
                'high_priority': [
                    '制定行为调整计划',
                    '练习快速决策',
                    '培养执行习惯'
                ],
                'medium_priority': [
                    '建立行为反馈机制',
                    '学习时间管理'
                ]
            },
            '情感调节': {
                'high_priority': [
                    '学习情绪管理技巧',
                    '练习压力释放方法',
                    '培养积极心态'
                ],
                'medium_priority': [
                    '建立支持网络',
                    '练习冥想放松'
                ]
            },
            '社会适应': {
                'high_priority': [
                    '提升沟通技巧',
                    '增强团队合作能力',
                    '扩大社交圈'
                ],
                'medium_priority': [
                    '学习冲突解决',
                    '培养领导力'
                ]
            },
            '技术学习': {
                'high_priority': [
                    '制定学习计划',
                    '练习新工具使用',
                    '参加技术培训'
                ],
                'medium_priority': [
                    '关注技术趋势',
                    '建立知识体系'
                ]
            },
            '战略思维': {
                'high_priority': [
                    '学习战略分析框架',
                    '练习长期规划',
                    '提升决策质量'
                ],
                'medium_priority': [
                    '关注行业动态',
                    '培养全局视野'
                ]
            }
        }
        
        priority = 'high_priority' if score < 0.4 else 'medium_priority'
        steps = suggestion_templates.get(dimension, {}).get(priority, ['通用提升方法'])
        
        return AdaptationSuggestion(
            suggestion_id=f"suggest_{len(self.suggestions)}",
            category=dimension,
            priority='高' if score < 0.4 else '中',
            description=f"{dimension}能力需要提升，当前评分: {score:.2f}",
            implementation_steps=steps,
            expected_improvement=0.2 if score < 0.4 else 0.1,
            time_estimate='2-4周' if score < 0.4 else '1-2周',
            resource_requirements=['时间投入', '学习资源', '实践机会']
        )
    
    def _generate_comprehensive_suggestion(self, metrics: AdaptabilityMetrics) -> AdaptationSuggestion:
        """生成综合建议"""
        lowest_dimension = min([
            ('认知灵活性', metrics.cognitive_flexibility),
            ('行为适应性', metrics.behavioral_adaptability),
            ('情感调节', metrics.emotional_regulation),
            ('社会适应', metrics.social_adaptation),
            ('技术学习', metrics.technical_learning),
            ('战略思维', metrics.strategic_thinking)
        ], key=lambda x: x[1])
        
        return AdaptationSuggestion(
            suggestion_id=f"comprehensive_{len(self.suggestions)}",
            category="综合提升",
            priority="高",
            description=f"总体适应能力需要系统性提升，重点关注{lowest_dimension[0]}",
            implementation_steps=[
                '制定系统性提升计划',
                f'重点训练{lowest_dimension[0]}',
                '建立定期评估机制',
                '寻求专业指导'
            ],
            expected_improvement=0.3,
            time_estimate='4-8周',
            resource_requirements=['专业指导', '系统训练', '实践机会', '时间投入']
        )
    
    def _calculate_development_trends(self, data: List[Dict]) -> Dict[str, Any]:
        """计算发展趋势"""
        if len(data) < 2:
            return {'overall': 'insufficient_data', 'dimensions': {}}
        
        # 提取指标数据
        scores = [item['metrics']['overall_score'] for item in data]
        
        # 计算趋势
        if len(scores) >= 3:
            recent_trend = np.polyfit(range(len(scores[-3:])), scores[-3:], 1)[0]
        else:
            recent_trend = scores[-1] - scores[0]
        
        overall_trend = 'improving' if recent_trend > 0.01 else 'declining' if recent_trend < -0.01 else 'stable'
        
        # 各维度趋势
        dimensions = ['cognitive_flexibility', 'behavioral_adaptability', 'emotional_regulation',
                     'social_adaptation', 'technical_learning', 'strategic_thinking']
        
        dimension_trends = {}
        for dim in dimensions:
            dim_scores = [item['metrics'][dim] for item in data]
            if len(dim_scores) >= 2:
                trend = dim_scores[-1] - dim_scores[0]
                dimension_trends[dim] = 'improving' if trend > 0.05 else 'declining' if trend < -0.05 else 'stable'
        
        return {
            'overall': overall_trend,
            'trend_value': recent_trend,
            'dimensions': dimension_trends
        }
    
    def _calculate_performance_metrics(self, data: List[Dict]) -> Dict[str, Any]:
        """计算性能指标"""
        if not data:
            return {}
        
        scores = [item['metrics']['overall_score'] for item in data]
        
        return {
            'average_score': np.mean(scores),
            'score_variance': np.var(scores),
            'best_score': max(scores),
            'worst_score': min(scores),
            'score_range': max(scores) - min(scores),
            'consistency': 1 - (np.var(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
        }
    
    def _identify_development_milestones(self, data: List[Dict]) -> List[Dict]:
        """识别发展里程碑"""
        milestones = []
        
        if len(data) < 2:
            return milestones
        
        scores = [item['metrics']['overall_score'] for item in data]
        
        # 寻找显著提升点
        for i in range(1, len(scores)):
            improvement = scores[i] - scores[i-1]
            if improvement > 0.1:  # 提升超过10%
                milestones.append({
                    'date': data[i]['timestamp'].isoformat(),
                    'type': 'significant_improvement',
                    'improvement': improvement,
                    'score': scores[i]
                })
        
        # 寻找最佳表现点
        best_index = np.argmax(scores)
        milestones.append({
            'date': data[best_index]['timestamp'].isoformat(),
            'type': 'peak_performance',
            'score': scores[best_index]
        })
        
        return milestones
    
    def _generate_development_recommendations(self, trends: Dict[str, Any]) -> List[str]:
        """生成发展建议"""
        recommendations = []
        
        if trends['overall'] == 'improving':
            recommendations.append("继续保持当前提升趋势")
        elif trends['overall'] == 'declining':
            recommendations.append("关注下降原因，调整策略")
        else:
            recommendations.append("考虑增加训练强度以实现突破")
        
        # 基于维度趋势的建议
        for dim, trend in trends['dimensions'].items():
            if trend == 'declining':
                recommendations.append(f"重点关注{dim}维度的提升")
        
        return recommendations
    
    def _calculate_improvement_progress(self) -> Dict[str, Any]:
        """计算改进进度"""
        if not self.model.historical_data:
            return {'error': '没有历史数据'}
        
        current = self.model.current_metrics.overall_score
        baseline = self.model.baseline_metrics.overall_score
        
        improvement = current - baseline
        progress_rate = improvement / (1 - baseline) if baseline < 1 else 0
        
        return {
            'baseline_score': baseline,
            'current_score': current,
            'absolute_improvement': improvement,
            'relative_improvement': progress_rate,
            'improvement_rate': 'positive' if improvement > 0 else 'negative' if improvement < 0 else 'stable'
        }
    
    def _analyze_environment_patterns(self) -> Dict[str, Any]:
        """分析环境模式"""
        if not self.environment_changes:
            return {'error': '没有环境变化数据'}
        
        change_types = [change.change_type for change in self.environment_changes]
        severities = [change.severity for change in self.environment_changes]
        
        return {
            'total_changes': len(self.environment_changes),
            'change_types': list(set(change_types)),
            'average_severity': np.mean(severities),
            'most_common_type': max(set(change_types), key=change_types.count) if change_types else 'unknown'
        }
    
    def _analyze_strategy_performance(self) -> Dict[str, Any]:
        """分析策略性能"""
        if not self.adaptation_performances:
            return {'error': '没有策略性能数据'}
        
        strategy_performance = {}
        for performance in self.adaptation_performances:
            strategy = performance.strategy_used.value
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(performance.outcome_score)
        
        analysis = {}
        for strategy, scores in strategy_performance.items():
            analysis[strategy] = {
                'average_performance': np.mean(scores),
                'performance_variance': np.var(scores),
                'usage_count': len(scores)
            }
        
        return analysis
    
    def _summarize_alerts(self) -> Dict[str, Any]:
        """总结预警"""
        if not self.alerts:
            return {'total_alerts': 0, 'recent_alerts': []}
        
        recent_alerts = [alert for alert in self.alerts 
                        if alert.get('timestamp', datetime.min) > datetime.now() - timedelta(days=7)]
        
        return {
            'total_alerts': len(self.alerts),
            'recent_alerts': recent_alerts[-5:],  # 最近5条预警
            'alert_types': list(set(alert.get('type', 'unknown') for alert in self.alerts))
        }
    
    def _analyze_development_trends(self) -> Dict[str, Any]:
        """分析发展趋势"""
        if not self.tracking_data:
            return {'error': '没有跟踪数据'}
        
        latest_tracking = self.tracking_data[-1]
        return latest_tracking.get('development_trends', {})
    
    def _generate_next_actions(self) -> List[str]:
        """生成下一步行动"""
        actions = []
        
        current_score = self.model.current_metrics.overall_score
        
        if current_score < 0.5:
            actions.append("立即开始系统性适应能力训练")
            actions.append("寻求专业指导和培训")
        elif current_score < 0.7:
            actions.append("制定针对性提升计划")
            actions.append("加强薄弱维度训练")
        else:
            actions.append("维持当前水平并分享经验")
            actions.append("挑战更高难度的适应任务")
        
        return actions


def main():
    """主函数 - 演示适应能力评估器功能"""
    print("=== D8 适应能力评估器演示 ===\n")
    
    # 创建评估器实例
    assessor = AdaptabilityAssessor()
    
    # 1. 构建适应能力模型
    print("1. 构建适应能力模型")
    initial_assessment = {
        'cognitive_flexibility': 0.75,
        'behavioral_adaptability': 0.68,
        'emotional_regulation': 0.72,
        'social_adaptation': 0.65,
        'technical_learning': 0.80,
        'strategic_thinking': 0.70
    }
    
    metrics = assessor.build_adaptability_model(initial_assessment)
    print(f"初始适应能力评分: {metrics.overall_score:.3f}\n")
    
    # 2. 环境变化适应性评估
    print("2. 环境变化适应性评估")
    env_result = assessor.assess_environment_adaptability(
        EnvironmentType.DYNAMIC, 0.7, 0.6
    )
    print(f"环境适应性评分: {env_result['overall_adaptability']:.3f}")
    print(f"建议: {env_result['recommendation']}\n")
    
    # 3. 适应速度和质量评估
    print("3. 适应速度和适应质量评估")
    adaptation_data = {
        'adaptation_time': 20,
        'expected_time': 24,
        'quality_indicators': {
            'accuracy': 0.85,
            'completeness': 0.80,
            'efficiency': 0.75,
            'sustainability': 0.78
        },
        'success_rate': 0.82,
        'resource_cost': 0.8
    }
    
    perf_result = assessor.assess_adaptation_speed_quality(adaptation_data)
    print(f"适应性能评分: {perf_result['overall_performance']:.3f}")
    print(f"速度评级: {perf_result['speed_rating']}")
    print(f"质量评级: {perf_result['quality_rating']}\n")
    
    # 4. 适应策略效果评估
    print("4. 适应策略效果评估")
    strategy_data = [
        {'success_rate': 0.8, 'efficiency_score': 0.75, 'resource_cost': 0.8},
        {'success_rate': 0.85, 'efficiency_score': 0.80, 'resource_cost': 0.75},
        {'success_rate': 0.78, 'efficiency_score': 0.72, 'resource_cost': 0.85}
    ]
    
    strategy_result = assessor.evaluate_strategy_effectiveness(
        AdaptationStrategy.PROACTIVE, strategy_data
    )
    print(f"策略效果评分: {strategy_result['effectiveness_score']:.3f}")
    print(f"平均成功率: {strategy_result['average_success_rate']:.3f}\n")
    
    # 5. 生成提升建议
    print("5. 适应能力提升建议")
    suggestions = assessor.generate_improvement_suggestions(metrics)
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"建议 {i}: {suggestion.description}")
        print(f"优先级: {suggestion.priority}")
        print(f"预期提升: {suggestion.expected_improvement:.2f}\n")
    
    # 6. 设置预警机制
    print("6. 设置适应能力预警机制")
    alert_config = {
        'low_adaptability_threshold': 0.6,
        'declining_performance_threshold': -0.05,
        'high_adaptation_time_threshold': 36,
        'low_success_rate_threshold': 0.7,
        'methods': ['log', 'notification']
    }
    
    alert_result = assessor.setup_adaptation_alerts(alert_config)
    print(f"预警机制状态: {alert_result['status']}")
    print(f"预警规则数量: {len(alert_result['alert_rules'])}\n")
    
    # 7. 跟踪发展
    print("7. 适应能力发展跟踪")
    tracking_result = assessor.track_adaptation_development(30)
    if 'error' not in tracking_result:
        print(f"数据点数: {tracking_result['data_points']}")
        print(f"总体趋势: {tracking_result['overall_trend']}")
        print(f"里程碑数量: {len(tracking_result['milestones'])}\n")
    
    # 8. 生成综合报告
    print("8. 生成综合评估报告")
    report = assessor.generate_comprehensive_report()
    print(f"报告ID: {report['report_id']}")
    print(f"当前总体评分: {report['current_metrics']['overall_score']:.3f}")
    print(f"改进进度: {report['improvement_progress']['absolute_improvement']:.3f}")
    print(f"建议数量: {len(report['recommendations'])}\n")
    
    print("=== 适应能力评估器演示完成 ===")


if __name__ == "__main__":
    main()