"""
D3能力边界评估器
实现多维度能力评估框架，包括能力边界识别、评估量化、差距分析、发展潜力评估等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')


class CapabilityType(Enum):
    """能力类型枚举"""
    TECHNICAL = "技术能力"
    COGNITIVE = "认知能力"
    CREATIVE = "创造能力"
    SOCIAL = "社交能力"
    EMOTIONAL = "情感能力"
    PHYSICAL = "身体能力"
    LEARNING = "学习能力"
    LEADERSHIP = "领导能力"
    PROBLEM_SOLVING = "问题解决能力"
    ADAPTABILITY = "适应能力"


class BoundaryType(Enum):
    """边界类型枚举"""
    HARD_BOUNDARY = "硬边界"  # 不可突破的边界
    SOFT_BOUNDARY = "软边界"  # 可调整的边界
    DYNAMIC_BOUNDARY = "动态边界"  # 随时间变化的边界
    ADAPTIVE_BOUNDARY = "自适应边界"  # 响应式边界


@dataclass
class CapabilityMetrics:
    """能力指标数据类"""
    capability_type: CapabilityType
    current_level: float  # 当前能力水平 (0-100)
    potential_level: float  # 潜在能力水平 (0-100)
    confidence: float  # 评估置信度 (0-1)
    measurement_date: str
    context: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)


@dataclass
class BoundaryDefinition:
    """边界定义数据类"""
    boundary_type: BoundaryType
    min_level: float
    max_level: float
    threshold: float
    flexibility: float  # 边界可调整性 (0-1)
    adaptation_rate: float  # 自适应速率
    context_dependencies: List[str] = field(default_factory=list)


@dataclass
class CapabilityGap:
    """能力差距数据类"""
    capability_type: CapabilityType
    current_level: float
    target_level: float
    gap_size: float
    gap_type: str  # "performance", "potential", "contextual"
    priority: int  # 优先级 (1-10)
    time_estimate: float  # 预估提升时间 (月)
    difficulty: float  # 提升难度 (0-1)


@dataclass
class DevelopmentPotential:
    """发展潜力数据类"""
    capability_type: CapabilityType
    potential_score: float  # 发展潜力分数 (0-100)
    learning_rate: float  # 学习速度
    peak_time: float  # 达到峰值时间
    sustainability: float  # 持续性
    risk_factors: List[str] = field(default_factory=list)
    enabling_factors: List[str] = field(default_factory=list)


class CapabilityBoundaryAssessor:
    """D3能力边界评估器"""
    
    def __init__(self, 
                 capability_dimensions: List[CapabilityType] = None,
                 boundary_config: Dict[CapabilityType, BoundaryDefinition] = None):
        """
        初始化能力边界评估器
        
        Args:
            capability_dimensions: 能力维度列表
            boundary_config: 边界配置字典
        """
        self.capability_dimensions = capability_dimensions or list(CapabilityType)
        self.boundary_config = boundary_config or self._default_boundary_config()
        
        # 数据存储
        self.metrics_history: Dict[CapabilityType, List[CapabilityMetrics]] = {
            dim: [] for dim in self.capability_dimensions
        }
        self.boundary_history: Dict[CapabilityType, List[BoundaryDefinition]] = {
            dim: [] for dim in self.capability_dimensions
        }
        
        # 机器学习模型
        self.models: Dict[CapabilityType, Any] = {}
        self.scalers: Dict[CapabilityType, StandardScaler] = {}
        self.anomaly_detectors: Dict[CapabilityType, IsolationForest] = {}
        
        # 评估参数
        self.assessment_weights = {
            'current_performance': 0.4,
            'potential_ability': 0.3,
            'learning_velocity': 0.2,
            'context_adaptation': 0.1
        }
        
        # 预警阈值
        self.warning_thresholds = {
            'capability_decline': 0.15,  # 能力下降15%触发预警
            'boundary_approach': 0.8,    # 接近边界80%触发预警
            'stagnation_period': 3       # 3个月无提升触发预警
        }
        
        self._initialize_models()
    
    def _default_boundary_config(self) -> Dict[CapabilityType, BoundaryDefinition]:
        """默认边界配置"""
        return {
            CapabilityType.TECHNICAL: BoundaryDefinition(
                boundary_type=BoundaryType.DYNAMIC_BOUNDARY,
                min_level=0, max_level=100, threshold=85,
                flexibility=0.7, adaptation_rate=0.1
            ),
            CapabilityType.COGNITIVE: BoundaryDefinition(
                boundary_type=BoundaryType.ADAPTIVE_BOUNDARY,
                min_level=0, max_level=100, threshold=80,
                flexibility=0.8, adaptation_rate=0.15
            ),
            CapabilityType.CREATIVE: BoundaryDefinition(
                boundary_type=BoundaryType.SOFT_BOUNDARY,
                min_level=0, max_level=100, threshold=75,
                flexibility=0.9, adaptation_rate=0.2
            ),
            CapabilityType.SOCIAL: BoundaryDefinition(
                boundary_type=BoundaryType.DYNAMIC_BOUNDARY,
                min_level=0, max_level=100, threshold=78,
                flexibility=0.6, adaptation_rate=0.12
            ),
            CapabilityType.EMOTIONAL: BoundaryDefinition(
                boundary_type=BoundaryType.ADAPTIVE_BOUNDARY,
                min_level=0, max_level=100, threshold=82,
                flexibility=0.7, adaptation_rate=0.08
            ),
            CapabilityType.PHYSICAL: BoundaryDefinition(
                boundary_type=BoundaryType.HARD_BOUNDARY,
                min_level=0, max_level=100, threshold=90,
                flexibility=0.3, adaptation_rate=0.05
            ),
            CapabilityType.LEARNING: BoundaryDefinition(
                boundary_type=BoundaryType.DYNAMIC_BOUNDARY,
                min_level=0, max_level=100, threshold=85,
                flexibility=0.8, adaptation_rate=0.18
            ),
            CapabilityType.PROBLEM_SOLVING: BoundaryDefinition(
                boundary_type=BoundaryType.ADAPTIVE_BOUNDARY,
                min_level=0, max_level=100, threshold=80,
                flexibility=0.75, adaptation_rate=0.14
            ),
            CapabilityType.ADAPTABILITY: BoundaryDefinition(
                boundary_type=BoundaryType.SOFT_BOUNDARY,
                min_level=0, max_level=100, threshold=77,
                flexibility=0.85, adaptation_rate=0.16
            )
        }
    
    def _initialize_models(self):
        """初始化机器学习模型"""
        for capability_type in self.capability_dimensions:
            # 能力预测模型
            self.models[capability_type] = MLPRegressor(
                hidden_layer_sizes=(50, 30),
                max_iter=500,
                random_state=42,
                alpha=0.01
            )
            
            # 数据标准化器
            self.scalers[capability_type] = StandardScaler()
            
            # 异常检测器
            self.anomaly_detectors[capability_type] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
    
    def assess_capability_boundary(self, 
                                 capability_type: CapabilityType,
                                 current_metrics: CapabilityMetrics,
                                 historical_data: List[CapabilityMetrics] = None) -> Dict[str, Any]:
        """
        评估特定能力类型的边界
        
        Args:
            capability_type: 能力类型
            current_metrics: 当前能力指标
            historical_data: 历史数据
            
        Returns:
            边界评估结果字典
        """
        if historical_data is None:
            historical_data = self.metrics_history[capability_type]
        
        # 边界定义
        boundary_def = self.boundary_config[capability_type]
        
        # 计算当前边界位置
        current_boundary = self._calculate_current_boundary(
            capability_type, current_metrics, historical_data
        )
        
        # 边界类型分析
        boundary_analysis = self._analyze_boundary_type(
            capability_type, boundary_def, current_metrics
        )
        
        # 边界稳定性评估
        stability_score = self._assess_boundary_stability(
            capability_type, historical_data
        )
        
        # 边界可突破性分析
        breakthrough_potential = self._analyze_breakthrough_potential(
            capability_type, current_metrics, historical_data
        )
        
        return {
            'capability_type': capability_type,
            'current_boundary': current_boundary,
            'boundary_definition': boundary_def,
            'boundary_analysis': boundary_analysis,
            'stability_score': stability_score,
            'breakthrough_potential': breakthrough_potential,
            'assessment_confidence': current_metrics.confidence,
            'recommendations': self._generate_boundary_recommendations(
                capability_type, current_boundary, boundary_analysis
            )
        }
    
    def _calculate_current_boundary(self, 
                                  capability_type: CapabilityType,
                                  current_metrics: CapabilityMetrics,
                                  historical_data: List[CapabilityMetrics]) -> Dict[str, float]:
        """计算当前边界位置"""
        if len(historical_data) < 2:
            # 基于当前指标估算边界
            base_boundary = current_metrics.current_level * 1.2
            return {
                'upper_bound': min(base_boundary, 100),
                'lower_bound': max(current_metrics.current_level * 0.8, 0),
                'optimal_range': (current_metrics.current_level * 0.9, 
                                current_metrics.current_level * 1.1)
            }
        
        # 基于历史数据趋势分析边界
        levels = [m.current_level for m in historical_data[-10:]]  # 最近10次测量
        trends = np.diff(levels)
        
        # 计算上界和下界
        upper_bound = np.percentile(levels, 95) + np.std(levels) * 0.5
        lower_bound = np.percentile(levels, 5) - np.std(levels) * 0.5
        
        # 考虑趋势调整
        trend_factor = np.mean(trends) if len(trends) > 0 else 0
        upper_bound += trend_factor * 5
        lower_bound += trend_factor * 3
        
        return {
            'upper_bound': min(max(upper_bound, 0), 100),
            'lower_bound': max(min(lower_bound, 100), 0),
            'optimal_range': (np.percentile(levels, 25), np.percentile(levels, 75)),
            'trend_factor': trend_factor
        }
    
    def _analyze_boundary_type(self, 
                             capability_type: CapabilityType,
                             boundary_def: BoundaryDefinition,
                             current_metrics: CapabilityMetrics) -> Dict[str, Any]:
        """分析边界类型特征"""
        analysis = {
            'boundary_type': boundary_def.boundary_type,
            'flexibility_score': boundary_def.flexibility,
            'adaptation_capability': boundary_def.adaptation_rate,
            'context_sensitivity': len(boundary_def.context_dependencies) / 10.0,
            'resistance_level': 1.0 - boundary_def.flexibility
        }
        
        # 基于能力类型调整分析
        if capability_type == CapabilityType.PHYSICAL:
            analysis['resistance_level'] *= 1.5  # 身体能力边界更稳定
            analysis['adaptation_capability'] *= 0.7
        
        elif capability_type == CapabilityType.CREATIVE:
            analysis['flexibility_score'] *= 1.3  # 创造能力边界更灵活
            analysis['adaptation_capability'] *= 1.2
        
        return analysis
    
    def _assess_boundary_stability(self, 
                                 capability_type: CapabilityType,
                                 historical_data: List[CapabilityMetrics]) -> float:
        """评估边界稳定性"""
        if len(historical_data) < 3:
            return 0.5  # 默认中等稳定性
        
        levels = [m.current_level for m in historical_data]
        
        # 计算变异系数
        mean_level = np.mean(levels)
        std_level = np.std(levels)
        cv = std_level / mean_level if mean_level > 0 else 1.0
        
        # 计算趋势一致性
        if len(levels) >= 4:
            recent_trend = np.polyfit(range(len(levels[-4:])), levels[-4:], 1)[0]
            overall_trend = np.polyfit(range(len(levels)), levels, 1)[0]
            trend_consistency = 1.0 - abs(recent_trend - overall_trend) / max(abs(overall_trend), 0.1)
        else:
            trend_consistency = 0.5
        
        # 综合稳定性分数
        stability_score = (1.0 - min(cv, 1.0)) * 0.6 + trend_consistency * 0.4
        return max(0.0, min(1.0, stability_score))
    
    def _analyze_breakthrough_potential(self, 
                                      capability_type: CapabilityType,
                                      current_metrics: CapabilityMetrics,
                                      historical_data: List[CapabilityMetrics]) -> Dict[str, float]:
        """分析边界突破潜力"""
        # 计算当前水平与边界的距离
        boundary_def = self.boundary_config[capability_type]
        distance_to_threshold = boundary_def.threshold - current_metrics.current_level
        distance_ratio = max(0, distance_to_threshold) / boundary_def.threshold
        
        # 基于历史数据计算提升潜力
        if len(historical_data) >= 3:
            recent_improvements = []
            for i in range(1, min(4, len(historical_data))):
                improvement = historical_data[-i].current_level - historical_data[-i-1].current_level
                recent_improvements.append(improvement)
            avg_improvement = np.mean(recent_improvements)
        else:
            avg_improvement = 0
        
        # 计算突破潜力
        potential_score = (
            (1.0 - distance_ratio) * 0.4 +  # 距离因子
            min(avg_improvement / 10.0, 1.0) * 0.3 +  # 提升速度因子
            boundary_def.flexibility * 0.3  # 边界灵活性因子
        )
        
        return {
            'breakthrough_potential': max(0.0, min(1.0, potential_score)),
            'distance_to_threshold': distance_to_threshold,
            'improvement_velocity': avg_improvement,
            'flexibility_factor': boundary_def.flexibility
        }
    
    def _generate_boundary_recommendations(self, 
                                         capability_type: CapabilityType,
                                         current_boundary: Dict[str, float],
                                         boundary_analysis: Dict[str, Any]) -> List[str]:
        """生成边界相关建议"""
        recommendations = []
        
        # 基于边界类型的建议
        if boundary_analysis['boundary_type'] == BoundaryType.HARD_BOUNDARY:
            recommendations.append("此能力存在生理或结构性限制，建议专注于优化现有水平")
        elif boundary_analysis['boundary_type'] == BoundaryType.SOFT_BOUNDARY:
            recommendations.append("此能力边界较灵活，通过适当训练有较大提升空间")
        
        # 基于灵活性的建议
        if boundary_analysis['flexibility_score'] > 0.8:
            recommendations.append("建议尝试新的学习方法或环境来突破当前边界")
        elif boundary_analysis['flexibility_score'] < 0.3:
            recommendations.append("建议采用渐进式训练方法，避免过度刺激")
        
        # 基于适应能力的建议
        if boundary_analysis['adaptation_capability'] > 0.15:
            recommendations.append("此能力适应性强，可以尝试多样化的训练方式")
        
        return recommendations
    
    def quantify_capability_level(self, 
                                capability_type: CapabilityType,
                                metrics: CapabilityMetrics,
                                context_factors: Dict[str, float] = None) -> Dict[str, Any]:
        """
        量化能力水平
        
        Args:
            capability_type: 能力类型
            metrics: 能力指标
            context_factors: 上下文因素
            
        Returns:
            能力量化结果
        """
        # 基础量化
        base_score = metrics.current_level
        
        # 置信度调整
        confidence_adjusted_score = base_score * (0.5 + 0.5 * metrics.confidence)
        
        # 上下文调整
        context_factor = 1.0
        if context_factors:
            context_factor = np.mean(list(context_factors.values()))
            context_factor = max(0.5, min(1.5, context_factor))  # 限制在合理范围
        
        adjusted_score = confidence_adjusted_score * context_factor
        
        # 标准化到0-100范围
        final_score = max(0, min(100, adjusted_score))
        
        # 计算量化置信度
        quantification_confidence = self._calculate_quantification_confidence(
            capability_type, metrics, context_factors
        )
        
        return {
            'capability_type': capability_type,
            'base_score': base_score,
            'confidence_adjusted_score': confidence_adjusted_score,
            'context_adjusted_score': adjusted_score,
            'final_score': final_score,
            'quantification_confidence': quantification_confidence,
            'context_factors': context_factors or {},
            'score_components': {
                'base_component': base_score / final_score if final_score > 0 else 0,
                'confidence_component': (0.5 + 0.5 * metrics.confidence),
                'context_component': context_factor
            }
        }
    
    def _calculate_quantification_confidence(self, 
                                           capability_type: CapabilityType,
                                           metrics: CapabilityMetrics,
                                           context_factors: Dict[str, float] = None) -> float:
        """计算量化置信度"""
        base_confidence = metrics.confidence
        
        # 基于历史数据调整
        historical_data = self.metrics_history[capability_type]
        if len(historical_data) >= 3:
            recent_consistency = self._calculate_recent_consistency(historical_data[-5:])
            consistency_factor = recent_consistency
        else:
            consistency_factor = 0.5
        
        # 基于上下文因子调整
        context_factor = 1.0
        if context_factors and len(context_factors) > 0:
            context_variance = np.var(list(context_factors.values()))
            context_factor = max(0.7, 1.0 - context_variance)
        
        final_confidence = base_confidence * 0.6 + consistency_factor * 0.3 + context_factor * 0.1
        return max(0.0, min(1.0, final_confidence))
    
    def _calculate_recent_consistency(self, recent_data: List[CapabilityMetrics]) -> float:
        """计算最近数据的一致性"""
        if len(recent_data) < 2:
            return 0.5
        
        levels = [m.current_level for m in recent_data]
        differences = np.abs(np.diff(levels))
        avg_difference = np.mean(differences)
        
        # 转换为一致性分数（差异越小，一致性越高）
        consistency = max(0, 1.0 - avg_difference / 10.0)
        return consistency
    
    def analyze_capability_gaps(self, 
                              current_metrics: Dict[CapabilityType, CapabilityMetrics],
                              target_metrics: Dict[CapabilityType, CapabilityMetrics] = None) -> List[CapabilityGap]:
        """
        分析能力差距
        
        Args:
            current_metrics: 当前能力指标
            target_metrics: 目标能力指标（可选）
            
        Returns:
            能力差距列表
        """
        gaps = []
        
        for capability_type in self.capability_dimensions:
            current = current_metrics.get(capability_type)
            if not current:
                continue
            
            # 确定目标水平
            if target_metrics and capability_type in target_metrics:
                target = target_metrics[capability_type]
                target_level = target.current_level
                gap_type = "performance"
            else:
                # 使用边界阈值作为目标
                boundary_def = self.boundary_config[capability_type]
                target_level = boundary_def.threshold
                gap_type = "potential"
            
            # 计算差距
            gap_size = max(0, target_level - current.current_level)
            
            if gap_size > 0:
                # 计算优先级（基于差距大小和重要性）
                priority = self._calculate_gap_priority(capability_type, gap_size, current)
                
                # 估算提升时间
                time_estimate = self._estimate_improvement_time(capability_type, gap_size, current)
                
                # 计算难度
                difficulty = self._calculate_improvement_difficulty(capability_type, gap_size, current)
                
                gap = CapabilityGap(
                    capability_type=capability_type,
                    current_level=current.current_level,
                    target_level=target_level,
                    gap_size=gap_size,
                    gap_type=gap_type,
                    priority=priority,
                    time_estimate=time_estimate,
                    difficulty=difficulty
                )
                gaps.append(gap)
        
        # 按优先级排序
        gaps.sort(key=lambda x: x.priority, reverse=True)
        return gaps
    
    def _calculate_gap_priority(self, 
                              capability_type: CapabilityType,
                              gap_size: float,
                              current_metrics: CapabilityMetrics) -> int:
        """计算差距优先级"""
        # 基础优先级（基于差距大小）
        base_priority = min(10, int(gap_size / 5) + 1)
        
        # 能力类型权重
        type_weights = {
            CapabilityType.TECHNICAL: 1.2,
            CapabilityType.LEARNING: 1.1,
            CapabilityType.PROBLEM_SOLVING: 1.15,
            CapabilityType.ADAPTABILITY: 1.0,
            CapabilityType.CREATIVE: 0.9,
            CapabilityType.SOCIAL: 0.95,
            CapabilityType.EMOTIONAL: 0.85,
            CapabilityType.PHYSICAL: 0.8,
            CapabilityType.COGNITIVE: 1.05,
            CapabilityType.LEADERSHIP: 1.0
        }
        
        weighted_priority = base_priority * type_weights.get(capability_type, 1.0)
        
        # 考虑当前水平（当前水平越低，优先级越高）
        if current_metrics.current_level < 30:
            weighted_priority *= 1.3
        elif current_metrics.current_level < 50:
            weighted_priority *= 1.1
        
        return min(10, max(1, int(weighted_priority)))
    
    def _estimate_improvement_time(self, 
                                 capability_type: CapabilityType,
                                 gap_size: float,
                                 current_metrics: CapabilityMetrics) -> float:
        """估算能力提升时间（月）"""
        # 基础提升速度（每月可提升的分数）
        base_rates = {
            CapabilityType.LEARNING: 8.0,
            CapabilityType.TECHNICAL: 6.0,
            CapabilityType.PROBLEM_SOLVING: 5.5,
            CapabilityType.COGNITIVE: 5.0,
            CapabilityType.ADAPTABILITY: 4.5,
            CapabilityType.CREATIVE: 4.0,
            CapabilityType.SOCIAL: 3.5,
            CapabilityType.EMOTIONAL: 3.0,
            CapabilityType.LEADERSHIP: 2.5,
            CapabilityType.PHYSICAL: 2.0
        }
        
        base_rate = base_rates.get(capability_type, 3.0)
        
        # 考虑当前水平（水平越低，提升越快）
        level_factor = max(0.5, 1.0 - current_metrics.current_level / 100)
        
        # 考虑学习速度
        learning_factor = current_metrics.confidence
        
        effective_rate = base_rate * level_factor * learning_factor
        
        # 估算时间
        time_estimate = gap_size / max(effective_rate, 0.5)
        
        # 添加缓冲时间
        time_estimate *= 1.2
        
        return max(1.0, time_estimate)
    
    def _calculate_improvement_difficulty(self, 
                                        capability_type: CapabilityType,
                                        gap_size: float,
                                        current_metrics: CapabilityMetrics) -> float:
        """计算提升难度"""
        # 基础难度
        base_difficulty = gap_size / 50.0  # 假设50分是中等差距
        
        # 能力类型难度系数
        type_difficulties = {
            CapabilityType.PHYSICAL: 1.5,
            CapabilityType.LEADERSHIP: 1.3,
            CapabilityType.EMOTIONAL: 1.2,
            CapabilityType.SOCIAL: 1.1,
            CapabilityType.CREATIVE: 1.0,
            CapabilityType.TECHNICAL: 0.9,
            CapabilityType.COGNITIVE: 0.85,
            CapabilityType.PROBLEM_SOLVING: 0.8,
            CapabilityType.ADAPTABILITY: 0.75,
            CapabilityType.LEARNING: 0.7
        }
        
        difficulty = base_difficulty * type_difficulties.get(capability_type, 1.0)
        
        # 考虑当前水平（水平越低，难度相对越小）
        if current_metrics.current_level < 30:
            difficulty *= 0.8
        elif current_metrics.current_level > 70:
            difficulty *= 1.2
        
        return max(0.1, min(1.0, difficulty))
    
    def assess_development_potential(self, 
                                   capability_type: CapabilityType,
                                   current_metrics: CapabilityMetrics,
                                   historical_data: List[CapabilityMetrics] = None) -> DevelopmentPotential:
        """
        评估能力发展潜力
        
        Args:
            capability_type: 能力类型
            current_metrics: 当前能力指标
            historical_data: 历史数据
            
        Returns:
            发展潜力评估结果
        """
        if historical_data is None:
            historical_data = self.metrics_history[capability_type]
        
        # 计算发展潜力分数
        potential_score = self._calculate_development_potential_score(
            capability_type, current_metrics, historical_data
        )
        
        # 计算学习速度
        learning_rate = self._calculate_learning_rate(historical_data)
        
        # 估算达到峰值时间
        peak_time = self._estimate_peak_time(capability_type, current_metrics, potential_score)
        
        # 评估持续性
        sustainability = self._assess_sustainability(capability_type, historical_data)
        
        # 识别风险因素
        risk_factors = self._identify_risk_factors(capability_type, current_metrics, historical_data)
        
        # 识别促进因素
        enabling_factors = self._identify_enabling_factors(capability_type, current_metrics, historical_data)
        
        return DevelopmentPotential(
            capability_type=capability_type,
            potential_score=potential_score,
            learning_rate=learning_rate,
            peak_time=peak_time,
            sustainability=sustainability,
            risk_factors=risk_factors,
            enabling_factors=enabling_factors
        )
    
    def _calculate_development_potential_score(self, 
                                             capability_type: CapabilityType,
                                             current_metrics: CapabilityMetrics,
                                             historical_data: List[CapabilityMetrics]) -> float:
        """计算发展潜力分数"""
        # 基础潜力（基于当前水平与边界的距离）
        boundary_def = self.boundary_config[capability_type]
        distance_to_boundary = boundary_def.threshold - current_metrics.current_level
        base_potential = max(0, distance_to_boundary) / boundary_def.threshold
        
        # 学习速度因子
        if len(historical_data) >= 3:
            recent_trend = self._calculate_recent_trend(historical_data[-5:])
            speed_factor = max(0, min(1.0, recent_trend / 5.0))  # 标准化到0-1
        else:
            speed_factor = 0.5
        
        # 稳定性因子
        stability_factor = 1.0 - abs(current_metrics.confidence - 0.5) * 2
        
        # 能力类型特性因子
        type_potential_factors = {
            CapabilityType.LEARNING: 1.2,
            CapabilityType.TECHNICAL: 1.1,
            CapabilityType.PROBLEM_SOLVING: 1.05,
            CapabilityType.ADAPTABILITY: 1.0,
            CapabilityType.CREATIVE: 0.95,
            CapabilityType.COGNITIVE: 0.9,
            CapabilityType.SOCIAL: 0.85,
            CapabilityType.EMOTIONAL: 0.8,
            CapabilityType.LEADERSHIP: 0.75,
            CapabilityType.PHYSICAL: 0.7
        }
        
        type_factor = type_potential_factors.get(capability_type, 1.0)
        
        # 综合潜力分数
        potential_score = (
            base_potential * 0.5 +
            speed_factor * 0.3 +
            stability_factor * 0.2
        ) * type_factor * 100
        
        return max(0, min(100, potential_score))
    
    def _calculate_recent_trend(self, recent_data: List[CapabilityMetrics]) -> float:
        """计算最近趋势"""
        if len(recent_data) < 2:
            return 0
        
        levels = [m.current_level for m in recent_data]
        if len(levels) >= 2:
            return levels[-1] - levels[0]
        return 0
    
    def _calculate_learning_rate(self, historical_data: List[CapabilityMetrics]) -> float:
        """计算学习速度"""
        if len(historical_data) < 3:
            return 0.5
        
        # 计算最近几次测量的改进率
        improvements = []
        for i in range(1, min(4, len(historical_data))):
            improvement = (historical_data[-i].current_level - 
                          historical_data[-i-1].current_level)
            improvements.append(improvement)
        
        avg_improvement = np.mean(improvements)
        
        # 标准化学习速度
        learning_rate = max(0, min(1.0, avg_improvement / 10.0))
        return learning_rate
    
    def _estimate_peak_time(self, 
                          capability_type: CapabilityType,
                          current_metrics: CapabilityMetrics,
                          potential_score: float) -> float:
        """估算达到峰值时间"""
        # 基础时间估算
        base_time = (100 - current_metrics.current_level) / 5.0  # 假设每月提升5分
        
        # 潜力调整
        potential_factor = potential_score / 100.0
        adjusted_time = base_time / max(potential_factor, 0.1)
        
        # 能力类型调整
        type_time_factors = {
            CapabilityType.PHYSICAL: 1.5,  # 身体能力需要更长时间
            CapabilityType.LEADERSHIP: 1.3,
            CapabilityType.EMOTIONAL: 1.2,
            CapabilityType.SOCIAL: 1.1,
            CapabilityType.CREATIVE: 1.0,
            CapabilityType.TECHNICAL: 0.9,
            CapabilityType.COGNITIVE: 0.85,
            CapabilityType.PROBLEM_SOLVING: 0.8,
            CapabilityType.ADAPTABILITY: 0.75,
            CapabilityType.LEARNING: 0.7
        }
        
        time_factor = type_time_factors.get(capability_type, 1.0)
        final_time = adjusted_time * time_factor
        
        return max(1.0, final_time)
    
    def _assess_sustainability(self, 
                             capability_type: CapabilityType,
                             historical_data: List[CapabilityMetrics]) -> float:
        """评估能力持续性"""
        if len(historical_data) < 5:
            return 0.5
        
        levels = [m.current_level for m in historical_data]
        
        # 计算趋势一致性
        trend_consistency = self._calculate_trend_consistency(levels)
        
        # 计算稳定性
        level_stability = 1.0 - (np.std(levels) / np.mean(levels))
        
        # 计算长期趋势
        if len(levels) >= 6:
            long_term_trend = np.polyfit(range(len(levels)), levels, 1)[0]
            sustainability = max(0, min(1.0, trend_consistency * 0.4 + 
                                     level_stability * 0.4 + 
                                     min(long_term_trend / 2.0, 1.0) * 0.2))
        else:
            sustainability = trend_consistency * 0.6 + level_stability * 0.4
        
        return max(0.0, min(1.0, sustainability))
    
    def _calculate_trend_consistency(self, levels: List[float]) -> float:
        """计算趋势一致性"""
        if len(levels) < 3:
            return 0.5
        
        # 计算相邻差异
        differences = np.diff(levels)
        
        # 计算趋势方向的一致性
        positive_diffs = sum(1 for d in differences if d > 0)
        negative_diffs = sum(1 for d in differences if d < 0)
        
        # 一致性分数
        consistency = max(positive_diffs, negative_diffs) / len(differences)
        return consistency
    
    def _identify_risk_factors(self, 
                             capability_type: CapabilityType,
                             current_metrics: CapabilityMetrics,
                             historical_data: List[CapabilityMetrics]) -> List[str]:
        """识别风险因素"""
        risk_factors = []
        
        # 基于当前水平的风险
        if current_metrics.current_level < 30:
            risk_factors.append("当前能力水平较低，可能存在基础薄弱风险")
        elif current_metrics.current_level > 85:
            risk_factors.append("接近能力边界，提升难度显著增加")
        
        # 基于历史数据的风险
        if len(historical_data) >= 3:
            recent_levels = [m.current_level for m in historical_data[-3:]]
            if np.std(recent_levels) > 10:
                risk_factors.append("能力水平波动较大，存在稳定性风险")
            
            # 检查下降趋势
            if recent_levels[-1] < recent_levels[0]:
                risk_factors.append("最近出现能力下降趋势")
        
        # 基于置信度的风险
        if current_metrics.confidence < 0.6:
            risk_factors.append("评估置信度较低，可能存在测量误差")
        
        # 能力类型特定风险
        type_specific_risks = {
            CapabilityType.PHYSICAL: "身体能力受年龄和健康状况影响较大",
            CapabilityType.EMOTIONAL: "情感能力易受环境和心理状态影响",
            CapabilityType.LEADERSHIP: "领导能力发展需要实践机会和反馈",
            CapabilityType.CREATIVE: "创造能力可能存在瓶颈期"
        }
        
        if capability_type in type_specific_risks:
            risk_factors.append(type_specific_risks[capability_type])
        
        return risk_factors
    
    def _identify_enabling_factors(self, 
                                 capability_type: CapabilityType,
                                 current_metrics: CapabilityMetrics,
                                 historical_data: List[CapabilityMetrics]) -> List[str]:
        """识别促进因素"""
        enabling_factors = []
        
        # 基于当前水平的促进因素
        if current_metrics.current_level > 60:
            enabling_factors.append("具备良好的能力基础，有利于进一步发展")
        
        if current_metrics.confidence > 0.8:
            enabling_factors.append("自我认知准确，有利于制定合适的发展策略")
        
        # 基于历史趋势的促进因素
        if len(historical_data) >= 3:
            recent_trend = self._calculate_recent_trend(historical_data[-3:])
            if recent_trend > 0:
                enabling_factors.append("呈现良好的上升趋势")
            
            # 检查稳定性
            recent_levels = [m.current_level for m in historical_data[-3:]]
            if np.std(recent_levels) < 5:
                enabling_factors.append("能力发展稳定，有利于持续提升")
        
        # 能力类型特定促进因素
        type_specific_enablers = {
            CapabilityType.LEARNING: "学习能力强，容易掌握新技能和方法",
            CapabilityType.TECHNICAL: "技术能力基础扎实，有利于深入发展",
            CapabilityType.ADAPTABILITY: "适应性强，能够应对各种挑战",
            CapabilityType.PROBLEM_SOLVING: "逻辑思维清晰，分析能力强",
            CapabilityType.CREATIVE: "创新思维活跃，容易产生新想法"
        }
        
        if capability_type in type_specific_enablers:
            enabling_factors.append(type_specific_enablers[capability_type])
        
        return enabling_factors
    
    def adjust_capability_boundaries(self, 
                                   capability_type: CapabilityType,
                                   new_metrics: CapabilityMetrics,
                                   adjustment_factors: Dict[str, float] = None) -> BoundaryDefinition:
        """
        动态调整能力边界
        
        Args:
            capability_type: 能力类型
            new_metrics: 新的能力指标
            adjustment_factors: 调整因子
            
        Returns:
            调整后的边界定义
        """
        current_boundary = self.boundary_config[capability_type]
        
        # 计算调整幅度
        adjustment_magnitude = self._calculate_adjustment_magnitude(
            capability_type, new_metrics, adjustment_factors
        )
        
        # 应用调整
        adjusted_boundary = self._apply_boundary_adjustment(
            current_boundary, adjustment_magnitude, new_metrics
        )
        
        # 更新边界配置
        self.boundary_config[capability_type] = adjusted_boundary
        
        # 记录边界历史
        if capability_type not in self.boundary_history:
            self.boundary_history[capability_type] = []
        self.boundary_history[capability_type].append(adjusted_boundary)
        
        return adjusted_boundary
    
    def _calculate_adjustment_magnitude(self, 
                                      capability_type: CapabilityType,
                                      new_metrics: CapabilityMetrics,
                                      adjustment_factors: Dict[str, float] = None) -> float:
        """计算调整幅度"""
        # 基础调整幅度
        base_adjustment = (new_metrics.current_level - 50) / 100.0  # 标准化到-0.5到0.5
        
        # 边界类型调整系数
        type_adjustment_factors = {
            BoundaryType.HARD_BOUNDARY: 0.1,
            BoundaryType.SOFT_BOUNDARY: 0.3,
            BoundaryType.DYNAMIC_BOUNDARY: 0.2,
            BoundaryType.ADAPTIVE_BOUNDARY: 0.25
        }
        
        boundary_type = self.boundary_config[capability_type].boundary_type
        type_factor = type_adjustment_factors.get(boundary_type, 0.2)
        
        # 应用调整因子
        factor_adjustment = 0
        if adjustment_factors:
            factor_adjustment = np.mean(list(adjustment_factors.values())) - 1.0
        
        # 置信度调整
        confidence_adjustment = (new_metrics.confidence - 0.5) * 0.2
        
        total_adjustment = (base_adjustment * type_factor + 
                          factor_adjustment * 0.5 + 
                          confidence_adjustment)
        
        return total_adjustment
    
    def _apply_boundary_adjustment(self, 
                                 current_boundary: BoundaryDefinition,
                                 adjustment_magnitude: float,
                                 new_metrics: CapabilityMetrics) -> BoundaryDefinition:
        """应用边界调整"""
        # 计算新的阈值
        new_threshold = current_boundary.threshold + adjustment_magnitude * 20
        new_threshold = max(0, min(100, new_threshold))
        
        # 计算新的边界范围
        range_expansion = abs(adjustment_magnitude) * 10
        new_min_level = max(0, current_boundary.min_level - range_expansion * 0.5)
        new_max_level = min(100, current_boundary.max_level + range_expansion * 0.5)
        
        # 调整灵活性和适应速率
        new_flexibility = current_boundary.flexibility + adjustment_magnitude * 0.1
        new_flexibility = max(0, min(1, new_flexibility))
        
        new_adaptation_rate = current_boundary.adaptation_rate + adjustment_magnitude * 0.05
        new_adaptation_rate = max(0, min(1, new_adaptation_rate))
        
        return BoundaryDefinition(
            boundary_type=current_boundary.boundary_type,
            min_level=new_min_level,
            max_level=new_max_level,
            threshold=new_threshold,
            flexibility=new_flexibility,
            adaptation_rate=new_adaptation_rate,
            context_dependencies=current_boundary.context_dependencies.copy()
        )
    
    def generate_early_warning(self, 
                             capability_type: CapabilityType,
                             current_metrics: CapabilityMetrics,
                             historical_data: List[CapabilityMetrics] = None) -> Dict[str, Any]:
        """
        生成能力边界预警
        
        Args:
            capability_type: 能力类型
            current_metrics: 当前能力指标
            historical_data: 历史数据
            
        Returns:
            预警信息字典
        """
        if historical_data is None:
            historical_data = self.metrics_history[capability_type]
        
        warnings = []
        warning_level = "normal"  # normal, caution, warning, critical
        
        # 检查能力下降预警
        decline_warning = self._check_capability_decline(
            capability_type, current_metrics, historical_data
        )
        if decline_warning:
            warnings.append(decline_warning)
            warning_level = max(warning_level, "caution", key=["normal", "caution", "warning", "critical"].index)
        
        # 检查边界接近预警
        boundary_warning = self._check_boundary_approach(
            capability_type, current_metrics
        )
        if boundary_warning:
            warnings.append(boundary_warning)
            warning_level = max(warning_level, "warning", key=["normal", "caution", "warning", "critical"].index)
        
        # 检查停滞预警
        stagnation_warning = self._check_stagnation_warning(
            capability_type, historical_data
        )
        if stagnation_warning:
            warnings.append(stagnation_warning)
            warning_level = max(warning_level, "caution", key=["normal", "caution", "warning", "critical"].index)
        
        # 检查异常模式预警
        anomaly_warning = self._check_anomaly_warning(
            capability_type, current_metrics, historical_data
        )
        if anomaly_warning:
            warnings.append(anomaly_warning)
            warning_level = max(warning_level, "critical", key=["normal", "caution", "warning", "critical"].index)
        
        return {
            'capability_type': capability_type,
            'warning_level': warning_level,
            'warnings': warnings,
            'timestamp': current_metrics.measurement_date,
            'recommendations': self._generate_warning_recommendations(warnings, warning_level)
        }
    
    def _check_capability_decline(self, 
                                capability_type: CapabilityType,
                                current_metrics: CapabilityMetrics,
                                historical_data: List[CapabilityMetrics]) -> Optional[Dict[str, Any]]:
        """检查能力下降预警"""
        if len(historical_data) < 2:
            return None
        
        # 计算下降幅度
        previous_level = historical_data[-1].current_level
        decline_rate = (previous_level - current_metrics.current_level) / previous_level
        
        if decline_rate > self.warning_thresholds['capability_decline']:
            return {
                'type': 'capability_decline',
                'severity': 'high' if decline_rate > 0.3 else 'medium',
                'decline_rate': decline_rate,
                'message': f"能力水平下降{decline_rate*100:.1f}%，需要关注"
            }
        return None
    
    def _check_boundary_approach(self, 
                               capability_type: CapabilityType,
                               current_metrics: CapabilityMetrics) -> Optional[Dict[str, Any]]:
        """检查边界接近预警"""
        boundary_def = self.boundary_config[capability_type]
        threshold_ratio = current_metrics.current_level / boundary_def.threshold
        
        if threshold_ratio > self.warning_thresholds['boundary_approach']:
            proximity = (threshold_ratio - self.warning_thresholds['boundary_approach']) / (1 - self.warning_thresholds['boundary_approach'])
            return {
                'type': 'boundary_approach',
                'severity': 'high' if proximity > 0.5 else 'medium',
                'proximity': proximity,
                'message': f"能力水平接近边界阈值，需要调整训练策略"
            }
        return None
    
    def _check_stagnation_warning(self, 
                                capability_type: CapabilityType,
                                historical_data: List[CapabilityMetrics]) -> Optional[Dict[str, Any]]:
        """检查能力停滞预警"""
        if len(historical_data) < self.warning_thresholds['stagnation_period']:
            return None
        
        # 检查最近几次测量是否有显著提升
        recent_levels = [m.current_level for m in historical_data[-self.warning_thresholds['stagnation_period']:]]
        max_improvement = max(recent_levels) - min(recent_levels)
        
        if max_improvement < 5:  # 5分以下认为停滞
            return {
                'type': 'stagnation',
                'severity': 'medium',
                'stagnation_period': self.warning_thresholds['stagnation_period'],
                'message': f"能力发展停滞{self.warning_thresholds['stagnation_period']}个测量周期，需要调整方法"
            }
        return None
    
    def _check_anomaly_warning(self, 
                             capability_type: CapabilityType,
                             current_metrics: CapabilityMetrics,
                             historical_data: List[CapabilityMetrics]) -> Optional[Dict[str, Any]]:
        """检查异常模式预警"""
        if len(historical_data) < 5:
            return None
        
        # 准备异常检测数据
        levels = [m.current_level for m in historical_data]
        current_level = current_metrics.current_level
        
        # 使用简单的统计方法检测异常
        mean_level = np.mean(levels)
        std_level = np.std(levels)
        
        # 计算Z-score
        if std_level > 0:
            z_score = abs(current_level - mean_level) / std_level
        else:
            z_score = 0
        
        # 检查是否为异常值（Z-score > 2）
        if z_score > 2.0:
            return {
                'type': 'anomaly',
                'severity': 'high',
                'anomaly_score': z_score,
                'message': "检测到异常能力模式，建议进行详细评估"
            }
        return None
    
    def _generate_warning_recommendations(self, 
                                        warnings: List[Dict[str, Any]], 
                                        warning_level: str) -> List[str]:
        """生成预警建议"""
        recommendations = []
        
        # 基于预警级别的建议
        if warning_level == "critical":
            recommendations.append("建议立即寻求专业指导和干预")
        elif warning_level == "warning":
            recommendations.append("建议调整当前训练方法，增加多样性")
        elif warning_level == "caution":
            recommendations.append("建议密切关注能力变化趋势")
        
        # 基于预警类型的具体建议
        warning_types = [w['type'] for w in warnings]
        
        if 'capability_decline' in warning_types:
            recommendations.append("分析下降原因，调整训练强度和方式")
            recommendations.append("增加休息和恢复时间")
        
        if 'boundary_approach' in warning_types:
            recommendations.append("考虑突破当前边界的策略")
            recommendations.append("探索新的学习方法和环境")
        
        if 'stagnation' in warning_types:
            recommendations.append("尝试新的训练方法或挑战")
            recommendations.append("寻求外部反馈和指导")
        
        if 'anomaly' in warning_types:
            recommendations.append("进行全面的能力评估")
            recommendations.append("排除健康或其他外部因素影响")
        
        return recommendations
    
    def generate_improvement_recommendations(self, 
                                           capability_type: CapabilityType,
                                           current_metrics: CapabilityMetrics,
                                           target_metrics: CapabilityMetrics = None,
                                           gaps: List[CapabilityGap] = None,
                                           potential: DevelopmentPotential = None) -> Dict[str, Any]:
        """
        生成能力提升建议
        
        Args:
            capability_type: 能力类型
            current_metrics: 当前能力指标
            target_metrics: 目标能力指标
            gaps: 能力差距列表
            potential: 发展潜力评估
            
        Returns:
            改进建议字典
        """
        # 获取相关信息
        if gaps is None:
            current_dict = {capability_type: current_metrics}
            target_dict = {capability_type: target_metrics} if target_metrics else None
            gaps = self.analyze_capability_gaps(current_dict, target_dict)
        
        if potential is None:
            potential = self.assess_development_potential(capability_type, current_metrics)
        
        # 生成基础建议
        basic_recommendations = self._generate_basic_recommendations(
            capability_type, current_metrics, gaps, potential
        )
        
        # 生成具体行动计划
        action_plan = self._generate_action_plan(
            capability_type, gaps, potential
        )
        
        # 生成资源配置建议
        resource_recommendations = self._generate_resource_recommendations(
            capability_type, gaps, potential
        )
        
        # 生成时间规划建议
        timeline_recommendations = self._generate_timeline_recommendations(
            gaps, potential
        )
        
        return {
            'capability_type': capability_type,
            'current_level': current_metrics.current_level,
            'target_level': target_metrics.current_level if target_metrics else None,
            'potential_score': potential.potential_score,
            'basic_recommendations': basic_recommendations,
            'action_plan': action_plan,
            'resource_recommendations': resource_recommendations,
            'timeline_recommendations': timeline_recommendations,
            'success_indicators': self._define_success_indicators(capability_type, gaps),
            'risk_mitigation': self._generate_risk_mitigation(capability_type, potential)
        }
    
    def _generate_basic_recommendations(self, 
                                      capability_type: CapabilityType,
                                      current_metrics: CapabilityMetrics,
                                      gaps: List[CapabilityGap],
                                      potential: DevelopmentPotential) -> List[str]:
        """生成基础建议"""
        recommendations = []
        
        # 基于当前水平的建议
        if current_metrics.current_level < 40:
            recommendations.append("建议从基础训练开始，夯实基本功")
        elif current_metrics.current_level < 70:
            recommendations.append("当前处于提升期，建议采用进阶训练方法")
        else:
            recommendations.append("已达到较高水平，建议专注精细化提升")
        
        # 基于发展潜力的建议
        if potential.potential_score > 80:
            recommendations.append("发展潜力巨大，建议制定雄心勃勃的提升目标")
        elif potential.potential_score > 60:
            recommendations.append("具备良好的发展潜力，建议持续投入")
        else:
            recommendations.append("发展潜力有限，建议专注优化现有水平")
        
        # 基于学习速度的建议
        if potential.learning_rate > 0.7:
            recommendations.append("学习能力强，可以尝试快速进阶训练")
        elif potential.learning_rate < 0.3:
            recommendations.append("建议采用循序渐进的方法，避免急于求成")
        
        # 基于能力类型的特定建议
        type_specific_recommendations = {
            CapabilityType.TECHNICAL: "建议通过项目实践来提升技术能力",
            CapabilityType.LEARNING: "建议学习高效学习方法和技巧",
            CapabilityType.CREATIVE: "建议多接触不同领域的知识和经验",
            CapabilityType.SOCIAL: "建议增加社交实践和团队合作机会",
            CapabilityType.EMOTIONAL: "建议进行情绪管理和心理健康训练",
            CapabilityType.LEADERSHIP: "建议承担更多责任和领导角色",
            CapabilityType.PROBLEM_SOLVING: "建议练习解决复杂问题的方法",
            CapabilityType.ADAPTABILITY: "建议主动面对变化和挑战",
            CapabilityType.COGNITIVE: "建议进行逻辑思维和认知训练",
            CapabilityType.PHYSICAL: "建议制定科学的训练和恢复计划"
        }
        
        if capability_type in type_specific_recommendations:
            recommendations.append(type_specific_recommendations[capability_type])
        
        return recommendations
    
    def _generate_action_plan(self, 
                            capability_type: CapabilityType,
                            gaps: List[CapabilityGap],
                            potential: DevelopmentPotential) -> List[Dict[str, Any]]:
        """生成具体行动计划"""
        action_plan = []
        
        # 基于差距的优先级行动
        for gap in gaps[:3]:  # 只考虑前3个最重要的差距
            action = {
                'priority': gap.priority,
                'objective': f"提升{capability_type.value}至{gap.target_level}分",
                'timeline': f"{gap.time_estimate:.1f}个月",
                'methods': self._suggest_improvement_methods(capability_type, gap),
                'milestones': self._define_milestones(gap),
                'success_criteria': self._define_success_criteria(gap)
            }
            action_plan.append(action)
        
        # 基于发展潜力的行动
        if potential.potential_score > 70:
            action_plan.append({
                'priority': 9,
                'objective': "探索能力边界突破机会",
                'timeline': "持续进行",
                'methods': ["尝试新的挑战", "寻求专业指导", "创造突破环境"],
                'milestones': ["每月评估边界变化", "记录突破尝试"],
                'success_criteria': ["能力水平超过历史最高值", "边界阈值有所提升"]
            })
        
        return sorted(action_plan, key=lambda x: x['priority'], reverse=True)
    
    def _suggest_improvement_methods(self, 
                                   capability_type: CapabilityType,
                                   gap: CapabilityGap) -> List[str]:
        """建议改进方法"""
        methods = []
        
        # 基于能力类型的改进方法
        type_methods = {
            CapabilityType.TECHNICAL: [
                "系统学习相关技术知识",
                "参与实际项目开发",
                "加入技术社区交流",
                "跟随行业专家学习"
            ],
            CapabilityType.LEARNING: [
                "学习高效学习技巧",
                "制定学习计划",
                "使用间隔重复方法",
                "寻求学习反馈"
            ],
            CapabilityType.CREATIVE: [
                "进行创意训练",
                "接触不同艺术形式",
                "参与头脑风暴",
                "学习创意思维方法"
            ],
            CapabilityType.SOCIAL: [
                "参与社交活动",
                "练习沟通技巧",
                "学习情商管理",
                "建立人际网络"
            ],
            CapabilityType.EMOTIONAL: [
                "进行情绪识别训练",
                "学习压力管理",
                "练习冥想和放松",
                "寻求心理指导"
            ],
            CapabilityType.LEADERSHIP: [
                "承担领导责任",
                "学习领导理论",
                "观察优秀领导者",
                "获取360度反馈"
            ],
            CapabilityType.PROBLEM_SOLVING: [
                "练习问题分析",
                "学习解决框架",
                "参与案例研究",
                "培养逻辑思维"
            ],
            CapabilityType.ADAPTABILITY: [
                "主动面对变化",
                "学习适应技巧",
                "培养开放心态",
                "练习快速决策"
            ],
            CapabilityType.COGNITIVE: [
                "进行认知训练",
                "练习逻辑推理",
                "学习批判性思维",
                "使用记忆技巧"
            ],
            CapabilityType.PHYSICAL: [
                "制定训练计划",
                "注意营养和休息",
                "寻求专业指导",
                "监控身体状况"
            ]
        }
        
        if capability_type in type_methods:
            methods.extend(type_methods[capability_type])
        
        # 基于差距大小的方法调整
        if gap.gap_size > 30:
            methods.append("考虑专业培训或指导")
        elif gap.gap_size < 10:
            methods.append("通过日常练习维持提升")
        
        # 基于难度的调整
        if gap.difficulty > 0.7:
            methods.append("分解目标，逐步实现")
            methods.append("寻求外部支持和指导")
        
        return methods
    
    def _define_milestones(self, gap: CapabilityGap) -> List[str]:
        """定义里程碑"""
        milestones = []
        
        # 分解目标为里程碑
        step_size = gap.gap_size / 4  # 分为4个阶段
        for i in range(1, 5):
            milestone_level = gap.current_level + step_size * i
            milestones.append(f"达到{gap.capability_type.value}{milestone_level:.1f}分")
        
        return milestones
    
    def _define_success_criteria(self, gap: CapabilityGap) -> List[str]:
        """定义成功标准"""
        criteria = [
            f"能力水平达到{gap.target_level}分",
            "能够稳定维持目标水平",
            "在相关任务中表现优秀"
        ]
        
        if gap.difficulty > 0.7:
            criteria.append("建立长期可持续的提升机制")
        
        return criteria
    
    def _generate_resource_recommendations(self, 
                                         capability_type: CapabilityType,
                                         gaps: List[CapabilityGap],
                                         potential: DevelopmentPotential) -> List[Dict[str, Any]]:
        """生成资源配置建议"""
        resources = []
        
        # 时间资源配置
        if potential.learning_rate > 0.6:
            time_allocation = "每周投入15-20小时进行专项训练"
        elif potential.learning_rate > 0.3:
            time_allocation = "每周投入10-15小时进行系统训练"
        else:
            time_allocation = "每周投入5-10小时进行稳定练习"
        
        resources.append({
            'type': '时间资源',
            'recommendation': time_allocation,
            'priority': 'high'
        })
        
        # 金钱资源配置
        cost_estimates = {
            CapabilityType.TECHNICAL: "每月500-2000元用于学习材料和课程",
            CapabilityType.LEARNING: "每月300-1000元用于学习工具和方法",
            CapabilityType.CREATIVE: "每月400-1500元用于材料和工具",
            CapabilityType.SOCIAL: "每月200-800元用于社交活动和网络建设",
            CapabilityType.EMOTIONAL: "每月500-2000元用于心理咨询和培训",
            CapabilityType.LEADERSHIP: "每月600-2500元用于培训和认证",
            CapabilityType.PROBLEM_SOLVING: "每月300-1200元用于培训和工具",
            CapabilityType.ADAPTABILITY: "每月200-800元用于培训和体验",
            CapabilityType.COGNITIVE: "每月400-1500元用于认知训练",
            CapabilityType.PHYSICAL: "每月800-3000元用于健身和医疗"
        }
        
        if capability_type in cost_estimates:
            resources.append({
                'type': '金钱资源',
                'recommendation': cost_estimates[capability_type],
                'priority': 'medium'
            })
        
        # 人力资源配置
        human_resources = []
        if potential.potential_score > 70:
            human_resources.append("寻找专业导师或教练")
        if gap := next((g for g in gaps if g.difficulty > 0.7), None):
            human_resources.append("加入学习小组或团队")
            human_resources.append("寻求同伴支持和反馈")
        
        if human_resources:
            resources.append({
                'type': '人力资源',
                'recommendation': human_resources,
                'priority': 'high'
            })
        
        # 技术资源
        tech_resources = {
            CapabilityType.TECHNICAL: "使用在线学习平台和开发工具",
            CapabilityType.LEARNING: "使用学习管理和记忆工具",
            CapabilityType.CREATIVE: "使用创意软件和协作平台",
            CapabilityType.SOCIAL: "使用社交网络和专业平台",
            CapabilityType.EMOTIONAL: "使用冥想和心理健康应用",
            CapabilityType.LEADERSHIP: "使用领导力评估和发展工具",
            CapabilityType.PROBLEM_SOLVING: "使用思维导图和分析工具",
            CapabilityType.ADAPTABILITY: "使用变化管理和适应训练工具",
            CapabilityType.COGNITIVE: "使用认知训练和大脑游戏",
            CapabilityType.PHYSICAL: "使用健身追踪和健康监测设备"
        }
        
        if capability_type in tech_resources:
            resources.append({
                'type': '技术资源',
                'recommendation': tech_resources[capability_type],
                'priority': 'medium'
            })
        
        return resources
    
    def _generate_timeline_recommendations(self, 
                                         gaps: List[CapabilityGap],
                                         potential: DevelopmentPotential) -> List[Dict[str, Any]]:
        """生成时间规划建议"""
        timeline = []
        
        # 短期目标（1-3个月）
        short_term_gaps = [g for g in gaps if g.time_estimate <= 3]
        if short_term_gaps:
            timeline.append({
                'period': '短期（1-3个月）',
                'objectives': [f"提升{g.capability_type.value}至{g.target_level}分" for g in short_term_gaps],
                'focus': '基础能力建设和习惯养成'
            })
        
        # 中期目标（3-12个月）
        medium_term_gaps = [g for g in gaps if 3 < g.time_estimate <= 12]
        if medium_term_gaps:
            timeline.append({
                'period': '中期（3-12个月）',
                'objectives': [f"达到{g.target_level}分水平" for g in medium_term_gaps],
                'focus': '能力深化和优化'
            })
        
        # 长期目标（1年以上）
        long_term_gaps = [g for g in gaps if g.time_estimate > 12]
        if long_term_gaps or potential.potential_score > 80:
            timeline.append({
                'period': '长期（1年以上）',
                'objectives': ['突破当前能力边界', '达到专家水平'],
                'focus': '边界突破和卓越追求'
            })
        
        # 基于持续性的建议
        if potential.sustainability < 0.5:
            timeline.append({
                'period': '持续关注',
                'objectives': ['建立可持续提升机制'],
                'focus': '长期稳定性维护'
            })
        
        return timeline
    
    def _define_success_indicators(self, 
                                 capability_type: CapabilityType,
                                 gaps: List[CapabilityGap]) -> List[str]:
        """定义成功指标"""
        indicators = []
        
        # 量化指标
        if gaps:
            main_gap = gaps[0]  # 最重要的差距
            indicators.append(f"能力水平从{main_gap.current_level:.1f}分提升至{main_gap.target_level:.1f}分")
            indicators.append(f"在{main_gap.time_estimate:.1f}个月内实现目标")
        
        # 质量指标
        indicators.extend([
            "能够稳定维持提升后的能力水平",
            "在相关任务中表现显著改善",
            "获得他人认可和积极反馈",
            "能够独立处理更复杂的挑战"
        ])
        
        # 能力类型特定指标
        type_indicators = {
            CapabilityType.TECHNICAL: "能够独立完成复杂技术项目",
            CapabilityType.LEARNING: "学习新技能的速度明显提升",
            CapabilityType.CREATIVE: "能够产出高质量的创新成果",
            CapabilityType.SOCIAL: "建立广泛有效的社交网络",
            CapabilityType.EMOTIONAL: "情绪管理能力显著提升",
            CapabilityType.LEADERSHIP: "成功领导团队完成重要项目",
            CapabilityType.PROBLEM_SOLVING: "能够快速有效解决复杂问题",
            CapabilityType.ADAPTABILITY: "能够快速适应新环境和变化",
            CapabilityType.COGNITIVE: "思维敏捷性和逻辑性明显提升",
            CapabilityType.PHYSICAL: "身体素质和健康状况显著改善"
        }
        
        if capability_type in type_indicators:
            indicators.append(type_indicators[capability_type])
        
        return indicators
    
    def _generate_risk_mitigation(self, 
                                capability_type: CapabilityType,
                                potential: DevelopmentPotential) -> List[Dict[str, Any]]:
        """生成风险缓解策略"""
        risk_mitigation = []
        
        # 基于风险因素的缓解策略
        for risk in potential.risk_factors:
            if "基础薄弱" in risk:
                risk_mitigation.append({
                    'risk': '基础薄弱风险',
                    'mitigation': '从基础训练开始，循序渐进',
                    'priority': 'high'
                })
            elif "瓶颈" in risk:
                risk_mitigation.append({
                    'risk': '发展瓶颈风险',
                    'mitigation': '尝试新的训练方法，寻求外部指导',
                    'priority': 'high'
                })
            elif "波动" in risk:
                risk_mitigation.append({
                    'risk': '能力波动风险',
                    'mitigation': '建立稳定的训练习惯，加强基础训练',
                    'priority': 'medium'
                })
        
        # 基于持续性的缓解策略
        if potential.sustainability < 0.6:
            risk_mitigation.append({
                'risk': '持续发展风险',
                'mitigation': '建立长期发展规划，注重可持续发展',
                'priority': 'high'
            })
        
        # 基于学习速度的缓解策略
        if potential.learning_rate < 0.4:
            risk_mitigation.append({
                'risk': '学习速度慢风险',
                'mitigation': '调整学习方法，寻求专业指导',
                'priority': 'medium'
            })
        
        # 通用风险缓解策略
        risk_mitigation.extend([
            {
                'risk': '动机不足风险',
                'mitigation': '设定阶段性目标，记录进步过程',
                'priority': 'medium'
            },
            {
                'risk': '时间不足风险',
                'mitigation': '制定现实可行的时间安排',
                'priority': 'medium'
            },
            {
                'risk': '资源不足风险',
                'mitigation': '寻找免费或低成本的替代资源',
                'priority': 'low'
            }
        ])
        
        return risk_mitigation
    
    def update_metrics_history(self, capability_type: CapabilityType, metrics: CapabilityMetrics):
        """更新能力指标历史记录"""
        if capability_type not in self.metrics_history:
            self.metrics_history[capability_type] = []
        
        self.metrics_history[capability_type].append(metrics)
        
        # 保持历史记录在合理范围内（最近50条）
        if len(self.metrics_history[capability_type]) > 50:
            self.metrics_history[capability_type] = self.metrics_history[capability_type][-50:]
    
    def get_comprehensive_assessment(self, 
                                   current_metrics: Dict[CapabilityType, CapabilityMetrics]) -> Dict[str, Any]:
        """
        获取综合能力评估报告
        
        Args:
            current_metrics: 当前能力指标字典
            
        Returns:
            综合评估报告
        """
        # 边界评估
        boundary_assessments = {}
        for capability_type, metrics in current_metrics.items():
            boundary_assessment = self.assess_capability_boundary(capability_type, metrics)
            boundary_assessments[capability_type.value] = boundary_assessment
        
        # 能力量化
        quantifications = {}
        for capability_type, metrics in current_metrics.items():
            quantification = self.quantify_capability_level(capability_type, metrics)
            quantifications[capability_type.value] = quantification
        
        # 差距分析
        gaps = self.analyze_capability_gaps(current_metrics)
        
        # 发展潜力评估
        potentials = {}
        for capability_type, metrics in current_metrics.items():
            potential = self.assess_development_potential(capability_type, metrics)
            potentials[capability_type.value] = potential
        
        # 预警检查
        warnings = {}
        for capability_type, metrics in current_metrics.items():
            warning = self.generate_early_warning(capability_type, metrics)
            if warning['warning_level'] != 'normal':
                warnings[capability_type.value] = warning
        
        # 改进建议
        improvement_recommendations = {}
        for capability_type, metrics in current_metrics.items():
            recommendation = self.generate_improvement_recommendations(capability_type, metrics)
            improvement_recommendations[capability_type.value] = recommendation
        
        # 综合分析
        overall_analysis = self._generate_overall_analysis(
            current_metrics, gaps, potentials, warnings
        )
        
        return {
            'assessment_date': current_metrics[list(current_metrics.keys())[0]].measurement_date,
            'boundary_assessments': boundary_assessments,
            'capability_quantifications': quantifications,
            'capability_gaps': [
                {
                    'capability_type': gap.capability_type.value,
                    'current_level': gap.current_level,
                    'target_level': gap.target_level,
                    'gap_size': gap.gap_size,
                    'priority': gap.priority,
                    'time_estimate': gap.time_estimate,
                    'difficulty': gap.difficulty
                } for gap in gaps
            ],
            'development_potentials': {
                k: {
                    'capability_type': v.capability_type.value,
                    'potential_score': v.potential_score,
                    'learning_rate': v.learning_rate,
                    'peak_time': v.peak_time,
                    'sustainability': v.sustainability
                } for k, v in potentials.items()
            },
            'early_warnings': warnings,
            'improvement_recommendations': improvement_recommendations,
            'overall_analysis': overall_analysis
        }
    
    def _generate_overall_analysis(self, 
                                 current_metrics: Dict[CapabilityType, CapabilityMetrics],
                                 gaps: List[CapabilityGap],
                                 potentials: Dict[CapabilityType, DevelopmentPotential],
                                 warnings: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合分析"""
        # 计算整体能力水平
        overall_level = np.mean([metrics.current_level for metrics in current_metrics.values()])
        
        # 计算整体发展潜力
        overall_potential = np.mean([potential.potential_score for potential in potentials.values()])
        
        # 识别关键能力领域
        key_capabilities = []
        sorted_gaps = sorted(gaps, key=lambda x: x.priority, reverse=True)
        for gap in sorted_gaps[:3]:  # 前3个最重要的差距
            key_capabilities.append({
                'capability': gap.capability_type.value,
                'gap_size': gap.gap_size,
                'priority': gap.priority
            })
        
        # 识别优势能力领域
        strength_capabilities = []
        for capability_type, metrics in current_metrics.items():
            if metrics.current_level > 75:
                strength_capabilities.append({
                    'capability': capability_type.value,
                    'level': metrics.current_level
                })
        
        # 评估整体风险级别
        risk_level = "low"
        if len(warnings) > 3:
            risk_level = "high"
        elif len(warnings) > 1:
            risk_level = "medium"
        
        # 生成综合建议
        overall_recommendations = []
        if overall_level < 50:
            overall_recommendations.append("建议优先提升基础能力水平")
        if overall_potential > 70:
            overall_recommendations.append("具备良好的整体发展潜力，建议制定全面的提升计划")
        if risk_level == "high":
            overall_recommendations.append("存在多个风险因素，建议寻求专业指导")
        
        return {
            'overall_capability_level': overall_level,
            'overall_development_potential': overall_potential,
            'key_capability_gaps': key_capabilities,
            'strength_capabilities': strength_capabilities,
            'overall_risk_level': risk_level,
            'total_capabilities_assessed': len(current_metrics),
            'total_warnings': len(warnings),
            'overall_recommendations': overall_recommendations
        }