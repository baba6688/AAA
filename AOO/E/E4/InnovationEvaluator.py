#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创新评估器 (InnovationEvaluator)

实现多维度创新评估框架，提供全面的创新性评估、质量评估、可行性分析、
影响评估、风险评估、优先级排序和建议改进功能。

主要功能：
1. 创新性评估和量化
2. 创新质量评估
3. 创新可行性分析
4. 创新影响评估
5. 创新风险评估
6. 创新优先级排序
7. 创新建议和改进


创建时间: 2025-11-05
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import json
import logging
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InnovationType(Enum):
    """创新类型枚举"""
    BREAKTHROUGH = "突破性创新"  # 颠覆性创新
    INCREMENTAL = "渐进性创新"   # 增量创新
    RADICAL = "根本性创新"      # 激进创新
    ARCHITECTURAL = "架构创新"   # 架构性创新
    MARKETING = "营销创新"      # 营销创新
    BUSINESS_MODEL = "商业模式创新"  # 商业模式创新


class RiskLevel(Enum):
    """风险等级枚举"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


class Priority(Enum):
    """优先级枚举"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    MINIMAL = 5


@dataclass
class InnovationMetrics:
    """创新评估指标数据结构"""
    # 创新性指标
    novelty_score: float = 0.0  # 新颖性得分 (0-100)
    originality_score: float = 0.0  # 原创性得分 (0-100)
    uniqueness_score: float = 0.0  # 独特性得分 (0-100)
    
    # 质量指标
    quality_score: float = 0.0  # 质量得分 (0-100)
    completeness_score: float = 0.0  # 完整性得分 (0-100)
    feasibility_score: float = 0.0  # 可行性得分 (0-100)
    
    # 影响指标
    impact_score: float = 0.0  # 影响得分 (0-100)
    market_potential: float = 0.0  # 市场潜力 (0-100)
    social_impact: float = 0.0  # 社会影响 (0-100)
    
    # 风险指标
    risk_score: float = 0.0  # 风险得分 (0-100)
    technical_risk: float = 0.0  # 技术风险 (0-100)
    market_risk: float = 0.0  # 市场风险 (0-100)
    
    # 综合指标
    overall_score: float = 0.0  # 综合得分 (0-100)
    priority_level: Priority = Priority.MEDIUM  # 优先级
    confidence_level: float = 0.0  # 置信度 (0-1)


@dataclass
class InnovationData:
    """创新项目数据结构"""
    project_id: str
    name: str
    description: str
    innovation_type: InnovationType
    category: str
    subcategory: str
    domain: str
    
    # 技术指标
    technology_readiness: int = 1  # 技术成熟度 (1-9)
    complexity_level: int = 1  # 复杂度等级 (1-5)
    resource_requirement: float = 0.0  # 资源需求 (0-1)
    
    # 市场指标
    market_size: float = 0.0  # 市场规模 (百万美元)
    competition_level: int = 1  # 竞争程度 (1-5)
    time_to_market: int = 12  # 上市时间 (月)
    
    # 团队指标
    team_size: int = 1  # 团队规模
    expertise_level: float = 0.0  # 专业知识水平 (0-1)
    experience_years: float = 0.0  # 经验年数
    
    # 历史数据
    success_rate: float = 0.0  # 历史成功率 (0-1)
    failure_rate: float = 0.0  # 历史失败率 (0-1)
    
    # 外部因素
    regulatory_support: float = 0.0  # 监管支持 (0-1)
    market_demand: float = 0.0  # 市场需求 (0-1)
    funding_availability: float = 0.0  # 资金可获得性 (0-1)


class InnovationEvaluator:
    """创新评估器主类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化创新评估器
        
        Args:
            config: 配置参数
        """
        self.config = config or self._default_config()
        self.metrics_weights = self._initialize_weights()
        self.ml_models = {}
        self.scalers = {}
        self.evaluation_history = []
        self.risk_thresholds = self._initialize_risk_thresholds()
        
        # 初始化机器学习模型
        self._initialize_ml_models()
        
        logger.info("创新评估器初始化完成")
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            "evaluation_dimensions": [
                "novelty", "quality", "feasibility", "impact", "risk"
            ],
            "ml_model_types": [
                "random_forest", "gradient_boosting", "linear_regression"
            ],
            "scaling_method": "minmax",
            "clustering_enabled": True,
            "real_time_updates": True,
            "confidence_threshold": 0.7
        }
    
    def _initialize_weights(self) -> Dict[str, float]:
        """初始化评估指标权重"""
        return {
            # 创新性权重
            "novelty": 0.25,
            "originality": 0.15,
            "uniqueness": 0.10,
            
            # 质量权重
            "quality": 0.20,
            "completeness": 0.10,
            "feasibility": 0.15,
            
            # 影响权重
            "impact": 0.20,
            "market_potential": 0.10,
            "social_impact": 0.05,
            
            # 风险权重 (负向影响)
            "risk": -0.15,
            "technical_risk": -0.08,
            "market_risk": -0.07
        }
    
    def _initialize_risk_thresholds(self) -> Dict[str, float]:
        """初始化风险阈值"""
        return {
            "very_low": 0.2,
            "low": 0.4,
            "medium": 0.6,
            "high": 0.8,
            "very_high": 1.0
        }
    
    def _initialize_ml_models(self):
        """初始化机器学习模型"""
        try:
            # 创新性预测模型
            self.ml_models["novelty_predictor"] = {
                "rf": RandomForestRegressor(n_estimators=100, random_state=42),
                "gb": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "lr": LinearRegression()
            }
            
            # 质量预测模型
            self.ml_models["quality_predictor"] = {
                "rf": RandomForestRegressor(n_estimators=100, random_state=42),
                "gb": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "lr": Ridge(alpha=1.0)
            }
            
            # 风险预测模型
            self.ml_models["risk_predictor"] = {
                "rf": RandomForestRegressor(n_estimators=100, random_state=42),
                "gb": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "lr": LinearRegression()
            }
            
            # 影响预测模型
            self.ml_models["impact_predictor"] = {
                "rf": RandomForestRegressor(n_estimators=100, random_state=42),
                "gb": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "lr": Ridge(alpha=1.0)
            }
            
            # 初始化数据缩放器
            self.scalers = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler()
            }
            
            logger.info("机器学习模型初始化完成")
            
        except Exception as e:
            logger.error(f"机器学习模型初始化失败: {e}")
    
    def evaluate_innovation(self, innovation_data: InnovationData) -> InnovationMetrics:
        """
        评估创新项目的综合指标
        
        Args:
            innovation_data: 创新项目数据
            
        Returns:
            InnovationMetrics: 评估结果
        """
        try:
            logger.info(f"开始评估创新项目: {innovation_data.name}")
            
            # 1. 创新性评估
            novelty_metrics = self._evaluate_novelty(innovation_data)
            
            # 2. 质量评估
            quality_metrics = self._evaluate_quality(innovation_data)
            
            # 3. 可行性分析
            feasibility_metrics = self._evaluate_feasibility(innovation_data)
            
            # 4. 影响评估
            impact_metrics = self._evaluate_impact(innovation_data)
            
            # 5. 风险评估
            risk_metrics = self._evaluate_risk(innovation_data)
            
            # 6. 计算综合得分
            overall_score = self._calculate_overall_score(
                novelty_metrics, quality_metrics, feasibility_metrics,
                impact_metrics, risk_metrics
            )
            
            # 7. 确定优先级
            priority = self._determine_priority(overall_score, risk_metrics)
            
            # 8. 计算置信度
            confidence = self._calculate_confidence(innovation_data)
            
            # 构建结果
            metrics = InnovationMetrics(
                novelty_score=novelty_metrics["overall"],
                originality_score=novelty_metrics["originality"],
                uniqueness_score=novelty_metrics["uniqueness"],
                quality_score=quality_metrics["overall"],
                completeness_score=quality_metrics["completeness"],
                feasibility_score=feasibility_metrics["overall"],
                impact_score=impact_metrics["overall"],
                market_potential=impact_metrics["market_potential"],
                social_impact=impact_metrics["social_impact"],
                risk_score=risk_metrics["overall"],
                technical_risk=risk_metrics["technical"],
                market_risk=risk_metrics["market"],
                overall_score=overall_score,
                priority_level=priority,
                confidence_level=confidence
            )
            
            # 保存评估历史
            self.evaluation_history.append({
                "timestamp": datetime.now(),
                "project_id": innovation_data.project_id,
                "metrics": metrics
            })
            
            logger.info(f"创新项目评估完成: {innovation_data.name}, 综合得分: {overall_score:.2f}")
            return metrics
            
        except Exception as e:
            logger.error(f"创新评估失败: {e}")
            raise
    
    def _evaluate_novelty(self, data: InnovationData) -> Dict[str, float]:
        """评估创新性指标"""
        try:
            # 基于历史数据和机器学习预测新颖性
            novelty_score = self._predict_novelty_ml(data)
            
            # 计算原创性得分
            originality_score = self._calculate_originality(data)
            
            # 计算独特性得分
            uniqueness_score = self._calculate_uniqueness(data)
            
            # 综合新颖性得分
            overall_novelty = (
                novelty_score * 0.5 +
                originality_score * 0.3 +
                uniqueness_score * 0.2
            )
            
            return {
                "novelty": novelty_score,
                "originality": originality_score,
                "uniqueness": uniqueness_score,
                "overall": overall_novelty
            }
            
        except Exception as e:
            logger.error(f"新颖性评估失败: {e}")
            return {"novelty": 50.0, "originality": 50.0, "uniqueness": 50.0, "overall": 50.0}
    
    def _evaluate_quality(self, data: InnovationData) -> Dict[str, float]:
        """评估质量指标"""
        try:
            # 技术质量评估
            tech_quality = self._assess_technical_quality(data)
            
            # 完整性评估
            completeness = self._assess_completeness(data)
            
            # 一致性评估
            consistency = self._assess_consistency(data)
            
            # 综合质量得分
            overall_quality = (
                tech_quality * 0.4 +
                completeness * 0.3 +
                consistency * 0.3
            )
            
            return {
                "technical": tech_quality,
                "completeness": completeness,
                "consistency": consistency,
                "overall": overall_quality
            }
            
        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            return {"technical": 50.0, "completeness": 50.0, "consistency": 50.0, "overall": 50.0}
    
    def _evaluate_feasibility(self, data: InnovationData) -> Dict[str, float]:
        """评估可行性指标"""
        try:
            # 技术可行性
            technical_feasibility = self._assess_technical_feasibility(data)
            
            # 经济可行性
            economic_feasibility = self._assess_economic_feasibility(data)
            
            # 运营可行性
            operational_feasibility = self._assess_operational_feasibility(data)
            
            # 时间可行性
            time_feasibility = self._assess_time_feasibility(data)
            
            # 综合可行性得分
            overall_feasibility = (
                technical_feasibility * 0.3 +
                economic_feasibility * 0.3 +
                operational_feasibility * 0.25 +
                time_feasibility * 0.15
            )
            
            return {
                "technical": technical_feasibility,
                "economic": economic_feasibility,
                "operational": operational_feasibility,
                "time": time_feasibility,
                "overall": overall_feasibility
            }
            
        except Exception as e:
            logger.error(f"可行性评估失败: {e}")
            return {
                "technical": 50.0, "economic": 50.0,
                "operational": 50.0, "time": 50.0, "overall": 50.0
            }
    
    def _evaluate_impact(self, data: InnovationData) -> Dict[str, float]:
        """评估影响指标"""
        try:
            # 市场影响
            market_impact = self._assess_market_impact(data)
            
            # 社会影响
            social_impact = self._assess_social_impact(data)
            
            # 技术影响
            tech_impact = self._assess_tech_impact(data)
            
            # 经济影响
            economic_impact = self._assess_economic_impact(data)
            
            # 市场潜力
            market_potential = self._calculate_market_potential(data)
            
            # 综合影响得分
            overall_impact = (
                market_impact * 0.3 +
                social_impact * 0.2 +
                tech_impact * 0.25 +
                economic_impact * 0.25
            )
            
            return {
                "market": market_impact,
                "social_impact": social_impact,
                "technical": tech_impact,
                "economic": economic_impact,
                "market_potential": market_potential,
                "overall": overall_impact
            }
            
        except Exception as e:
            logger.error(f"影响评估失败: {e}")
            return {
                "market": 50.0, "social_impact": 50.0,
                "technical": 50.0, "economic": 50.0,
                "market_potential": 50.0, "overall": 50.0
            }
    
    def _evaluate_risk(self, data: InnovationData) -> Dict[str, float]:
        """评估风险指标"""
        try:
            # 技术风险
            technical_risk = self._assess_technical_risk(data)
            
            # 市场风险
            market_risk = self._assess_market_risk(data)
            
            # 财务风险
            financial_risk = self._assess_financial_risk(data)
            
            # 运营风险
            operational_risk = self._assess_operational_risk(data)
            
            # 监管风险
            regulatory_risk = self._assess_regulatory_risk(data)
            
            # 综合风险得分 (转换为0-100量表，100表示最高风险)
            overall_risk = (
                technical_risk * 0.25 +
                market_risk * 0.25 +
                financial_risk * 0.2 +
                operational_risk * 0.15 +
                regulatory_risk * 0.15
            )
            
            return {
                "technical": technical_risk,
                "market": market_risk,
                "financial": financial_risk,
                "operational": operational_risk,
                "regulatory": regulatory_risk,
                "overall": overall_risk
            }
            
        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            return {
                "technical": 50.0, "market": 50.0,
                "financial": 50.0, "operational": 50.0,
                "regulatory": 50.0, "overall": 50.0
            }
    
    def _predict_novelty_ml(self, data: InnovationData) -> float:
        """使用机器学习预测新颖性"""
        try:
            # 准备特征向量
            features = self._prepare_features(data)
            
            # 使用随机森林预测
            if "novelty_predictor" in self.ml_models:
                rf_model = self.ml_models["novelty_predictor"]["rf"]
                # 假设有训练数据，这里返回基于规则的预测
                novelty_score = self._rule_based_novelty_prediction(data)
                return max(0, min(100, novelty_score))
            else:
                return self._rule_based_novelty_prediction(data)
                
        except Exception as e:
            logger.error(f"ML新颖性预测失败: {e}")
            return 50.0
    
    def _rule_based_novelty_prediction(self, data: InnovationData) -> float:
        """基于规则的新颖性预测"""
        base_score = 50.0
        
        # 根据创新类型调整
        type_multipliers = {
            InnovationType.BREAKTHROUGH: 1.5,
            InnovationType.RADICAL: 1.3,
            InnovationType.ARCHITECTURAL: 1.2,
            InnovationType.INCREMENTAL: 0.8,
            InnovationType.MARKETING: 0.9,
            InnovationType.BUSINESS_MODEL: 1.1
        }
        
        multiplier = type_multipliers.get(data.innovation_type, 1.0)
        
        # 根据技术成熟度调整
        maturity_factor = max(0.1, 1.0 - (data.technology_readiness - 1) * 0.1)
        
        # 根据复杂度调整
        complexity_factor = min(1.2, 0.8 + data.complexity_level * 0.1)
        
        novelty_score = base_score * multiplier * maturity_factor * complexity_factor
        return max(0, min(100, novelty_score))
    
    def _calculate_originality(self, data: InnovationData) -> float:
        """计算原创性得分"""
        # 基于跨领域特征计算原创性
        domain_combination_score = self._evaluate_domain_combination(data)
        
        # 基于技术融合程度计算原创性
        tech_fusion_score = self._evaluate_tech_fusion(data)
        
        # 基于概念新颖性计算原创性
        concept_novelty = self._evaluate_concept_novelty(data)
        
        originality = (
            domain_combination_score * 0.4 +
            tech_fusion_score * 0.3 +
            concept_novelty * 0.3
        )
        
        return max(0, min(100, originality))
    
    def _calculate_uniqueness(self, data: InnovationData) -> float:
        """计算独特性得分"""
        # 基于竞争分析
        competition_uniqueness = 100 - (data.competition_level * 20)
        
        # 基于技术独特性
        tech_uniqueness = self._evaluate_tech_uniqueness(data)
        
        # 基于应用独特性
        application_uniqueness = self._evaluate_application_uniqueness(data)
        
        uniqueness = (
            competition_uniqueness * 0.4 +
            tech_uniqueness * 0.35 +
            application_uniqueness * 0.25
        )
        
        return max(0, min(100, uniqueness))
    
    def _assess_technical_quality(self, data: InnovationData) -> float:
        """评估技术质量"""
        # 基于技术成熟度
        maturity_score = data.technology_readiness * 10
        
        # 基于团队专业水平
        expertise_score = data.expertise_level * 100
        
        # 基于技术复杂度适配度
        complexity_score = max(0, 100 - abs(data.complexity_level - 3) * 15)
        
        tech_quality = (
            maturity_score * 0.4 +
            expertise_score * 0.35 +
            complexity_score * 0.25
        )
        
        return max(0, min(100, tech_quality))
    
    def _assess_completeness(self, data: InnovationData) -> float:
        """评估完整性"""
        # 基于需求明确性
        requirement_clarity = self._evaluate_requirement_clarity(data)
        
        # 基于解决方案完整性
        solution_completeness = self._evaluate_solution_completeness(data)
        
        # 基于实施计划完整性
        plan_completeness = self._evaluate_plan_completeness(data)
        
        completeness = (
            requirement_clarity * 0.3 +
            solution_completeness * 0.4 +
            plan_completeness * 0.3
        )
        
        return max(0, min(100, completeness))
    
    def _assess_consistency(self, data: InnovationData) -> float:
        """评估一致性"""
        # 基于目标一致性
        goal_consistency = self._evaluate_goal_consistency(data)
        
        # 基于技术一致性
        tech_consistency = self._evaluate_tech_consistency(data)
        
        # 基于资源一致性
        resource_consistency = self._evaluate_resource_consistency(data)
        
        consistency = (
            goal_consistency * 0.4 +
            tech_consistency * 0.35 +
            resource_consistency * 0.25
        )
        
        return max(0, min(100, consistency))
    
    def _assess_technical_feasibility(self, data: InnovationData) -> float:
        """评估技术可行性"""
        # 基于技术成熟度
        maturity_feasibility = data.technology_readiness * 10
        
        # 基于团队技术能力
        team_capability = data.expertise_level * 100
        
        # 基于技术复杂度
        complexity_factor = max(0, 100 - data.complexity_level * 15)
        
        # 基于技术资源可获得性
        resource_factor = data.funding_availability * 100
        
        feasibility = (
            maturity_feasibility * 0.3 +
            team_capability * 0.3 +
            complexity_factor * 0.25 +
            resource_factor * 0.15
        )
        
        return max(0, min(100, feasibility))
    
    def _assess_economic_feasibility(self, data: InnovationData) -> float:
        """评估经济可行性"""
        # 基于市场潜力
        market_factor = min(100, data.market_size / 10)  # 假设每1000万美元为1分
        
        # 基于投资回报预期
        roi_factor = self._estimate_roi(data)
        
        # 基于成本效益比
        cost_benefit_ratio = self._calculate_cost_benefit_ratio(data)
        
        # 基于资金可获得性
        funding_factor = data.funding_availability * 100
        
        feasibility = (
            market_factor * 0.3 +
            roi_factor * 0.3 +
            cost_benefit_ratio * 0.25 +
            funding_factor * 0.15
        )
        
        return max(0, min(100, feasibility))
    
    def _assess_operational_feasibility(self, data: InnovationData) -> float:
        """评估运营可行性"""
        # 基于团队规模适配性
        team_size_factor = self._evaluate_team_size_adequacy(data)
        
        # 基于组织支持度
        org_support = self._evaluate_organizational_support(data)
        
        # 基于供应链准备度
        supply_chain_readiness = self._evaluate_supply_chain_readiness(data)
        
        # 基于运营流程成熟度
        process_maturity = self._evaluate_process_maturity(data)
        
        feasibility = (
            team_size_factor * 0.25 +
            org_support * 0.3 +
            supply_chain_readiness * 0.25 +
            process_maturity * 0.2
        )
        
        return max(0, min(100, feasibility))
    
    def _assess_time_feasibility(self, data: InnovationData) -> float:
        """评估时间可行性"""
        # 基于上市时间合理性
        time_to_market_factor = max(0, 100 - data.time_to_market * 2)
        
        # 基于项目复杂度时间适配性
        complexity_time_factor = max(0, 100 - data.complexity_level * 10)
        
        # 基于资源投入时间充足性
        resource_time_factor = min(100, data.resource_requirement * 100 * 2)
        
        # 基于团队经验时间适配性
        experience_time_factor = min(100, data.experience_years * 10)
        
        feasibility = (
            time_to_market_factor * 0.3 +
            complexity_time_factor * 0.25 +
            resource_time_factor * 0.25 +
            experience_time_factor * 0.2
        )
        
        return max(0, min(100, feasibility))
    
    def _assess_market_impact(self, data: InnovationData) -> float:
        """评估市场影响"""
        # 基于市场规模
        market_size_factor = min(100, data.market_size / 5)
        
        # 基于市场竞争度
        competition_factor = max(0, 100 - data.competition_level * 15)
        
        # 基于市场需求强度
        demand_factor = data.market_demand * 100
        
        # 基于市场时机
        timing_factor = self._evaluate_market_timing(data)
        
        impact = (
            market_size_factor * 0.3 +
            competition_factor * 0.25 +
            demand_factor * 0.3 +
            timing_factor * 0.15
        )
        
        return max(0, min(100, impact))
    
    def _assess_social_impact(self, data: InnovationData) -> float:
        """评估社会影响"""
        # 基于社会需求匹配度
        social_need_match = self._evaluate_social_need_match(data)
        
        # 基于社会接受度
        social_acceptance = self._evaluate_social_acceptance(data)
        
        # 基于环境影响
        environmental_impact = self._evaluate_environmental_impact(data)
        
        # 基于社会价值创造
        social_value_creation = self._evaluate_social_value_creation(data)
        
        impact = (
            social_need_match * 0.3 +
            social_acceptance * 0.25 +
            environmental_impact * 0.25 +
            social_value_creation * 0.2
        )
        
        return max(0, min(100, impact))
    
    def _assess_tech_impact(self, data: InnovationData) -> float:
        """评估技术影响"""
        # 基于技术先进性
        tech_advancement = self._evaluate_tech_advancement(data)
        
        # 基于技术扩散潜力
        tech_diffusion_potential = self._evaluate_tech_diffusion_potential(data)
        
        # 基于技术生态影响
        tech_ecosystem_impact = self._evaluate_tech_ecosystem_impact(data)
        
        # 基于技术标准化潜力
        tech_standardization_potential = self._evaluate_tech_standardization_potential(data)
        
        impact = (
            tech_advancement * 0.3 +
            tech_diffusion_potential * 0.25 +
            tech_ecosystem_impact * 0.25 +
            tech_standardization_potential * 0.2
        )
        
        return max(0, min(100, impact))
    
    def _assess_economic_impact(self, data: InnovationData) -> float:
        """评估经济影响"""
        # 基于经济价值创造
        economic_value_creation = self._estimate_economic_value_creation(data)
        
        # 基于就业创造潜力
        job_creation_potential = self._estimate_job_creation_potential(data)
        
        # 基于产业升级贡献
        industry_upgrade_contribution = self._evaluate_industry_upgrade_contribution(data)
        
        # 基于国家竞争力提升
        national_competitiveness_boost = self._evaluate_national_competitiveness_boost(data)
        
        impact = (
            economic_value_creation * 0.35 +
            job_creation_potential * 0.25 +
            industry_upgrade_contribution * 0.25 +
            national_competitiveness_boost * 0.15
        )
        
        return max(0, min(100, impact))
    
    def _calculate_market_potential(self, data: InnovationData) -> float:
        """计算市场潜力"""
        # 基于市场规模
        market_size_score = min(100, data.market_size / 2)
        
        # 基于市场增长率
        market_growth_rate = self._estimate_market_growth_rate(data)
        growth_score = min(100, market_growth_rate * 10)
        
        # 基于市场渗透潜力
        penetration_potential = self._estimate_market_penetration_potential(data)
        
        # 基于竞争优势
        competitive_advantage = self._assess_competitive_advantage(data)
        
        potential = (
            market_size_score * 0.3 +
            growth_score * 0.25 +
            penetration_potential * 0.25 +
            competitive_advantage * 0.2
        )
        
        return max(0, min(100, potential))
    
    def _assess_technical_risk(self, data: InnovationData) -> float:
        """评估技术风险"""
        # 基于技术成熟度风险
        maturity_risk = max(0, 100 - data.technology_readiness * 10)
        
        # 基于技术复杂度风险
        complexity_risk = data.complexity_level * 15
        
        # 基于技术团队风险
        team_risk = max(0, 100 - data.expertise_level * 100)
        
        # 基于技术依赖风险
        dependency_risk = self._assess_tech_dependency_risk(data)
        
        risk = (
            maturity_risk * 0.3 +
            complexity_risk * 0.3 +
            team_risk * 0.25 +
            dependency_risk * 0.15
        )
        
        return max(0, min(100, risk))
    
    def _assess_market_risk(self, data: InnovationData) -> float:
        """评估市场风险"""
        # 基于市场竞争风险
        competition_risk = data.competition_level * 15
        
        # 基于市场接受度风险
        acceptance_risk = max(0, 100 - data.market_demand * 100)
        
        # 基于市场变化风险
        market_change_risk = self._assess_market_change_risk(data)
        
        # 基于监管变化风险
        regulatory_change_risk = max(0, 100 - data.regulatory_support * 100)
        
        risk = (
            competition_risk * 0.3 +
            acceptance_risk * 0.3 +
            market_change_risk * 0.25 +
            regulatory_change_risk * 0.15
        )
        
        return max(0, min(100, risk))
    
    def _assess_financial_risk(self, data: InnovationData) -> float:
        """评估财务风险"""
        # 基于资金需求风险
        funding_risk = max(0, 100 - data.funding_availability * 100)
        
        # 基于投资回报不确定性
        roi_uncertainty = self._assess_roi_uncertainty(data)
        
        # 基于成本超支风险
        cost_overrun_risk = self._assess_cost_overrun_risk(data)
        
        # 基于现金流风险
        cash_flow_risk = self._assess_cash_flow_risk(data)
        
        risk = (
            funding_risk * 0.35 +
            roi_uncertainty * 0.25 +
            cost_overrun_risk * 0.25 +
            cash_flow_risk * 0.15
        )
        
        return max(0, min(100, risk))
    
    def _assess_operational_risk(self, data: InnovationData) -> float:
        """评估运营风险"""
        # 基于团队规模风险
        team_size_risk = abs(data.team_size - 5) * 5  # 假设5人为最优规模
        
        # 基于运营复杂度风险
        operational_complexity_risk = data.complexity_level * 12
        
        # 基于供应链风险
        supply_chain_risk = self._assess_supply_chain_risk(data)
        
        # 基于质量控制风险
        quality_control_risk = self._assess_quality_control_risk(data)
        
        risk = (
            team_size_risk * 0.25 +
            operational_complexity_risk * 0.3 +
            supply_chain_risk * 0.25 +
            quality_control_risk * 0.2
        )
        
        return max(0, min(100, risk))
    
    def _assess_regulatory_risk(self, data: InnovationData) -> float:
        """评估监管风险"""
        # 基于监管支持度
        regulatory_support_risk = max(0, 100 - data.regulatory_support * 100)
        
        # 基于合规复杂度
        compliance_complexity = self._assess_compliance_complexity(data)
        
        # 基于政策变化风险
        policy_change_risk = self._assess_policy_change_risk(data)
        
        # 基于国际监管风险
        international_regulatory_risk = self._assess_international_regulatory_risk(data)
        
        risk = (
            regulatory_support_risk * 0.3 +
            compliance_complexity * 0.25 +
            policy_change_risk * 0.25 +
            international_regulatory_risk * 0.2
        )
        
        return max(0, min(100, risk))
    
    def _calculate_overall_score(self, novelty: Dict, quality: Dict, 
                               feasibility: Dict, impact: Dict, risk: Dict) -> float:
        """计算综合得分"""
        try:
            # 加权计算各项得分
            novelty_score = novelty["overall"] * self.metrics_weights["novelty"]
            quality_score = quality["overall"] * self.metrics_weights["quality"]
            feasibility_score = feasibility["overall"] * self.metrics_weights["feasibility"]
            impact_score = impact["overall"] * self.metrics_weights["impact"]
            risk_score = risk["overall"] * self.metrics_weights["risk"]
            
            # 计算综合得分
            overall_score = (
                novelty_score +
                quality_score +
                feasibility_score +
                impact_score +
                risk_score
            )
            
            # 转换为0-100量表
            normalized_score = (overall_score + 50) * 10  # 假设原始得分范围为-50到50
            
            return max(0, min(100, normalized_score))
            
        except Exception as e:
            logger.error(f"综合得分计算失败: {e}")
            return 50.0
    
    def _determine_priority(self, overall_score: float, risk_metrics: Dict) -> Priority:
        """确定优先级"""
        try:
            # 综合得分权重
            score_weight = 0.7
            
            # 风险权重 (风险越低优先级越高)
            risk_weight = 0.3
            risk_factor = max(0, 100 - risk_metrics["overall"])
            
            # 计算综合优先级分数
            priority_score = overall_score * score_weight + risk_factor * risk_weight
            
            # 根据分数确定优先级
            if priority_score >= 80:
                return Priority.CRITICAL
            elif priority_score >= 65:
                return Priority.HIGH
            elif priority_score >= 50:
                return Priority.MEDIUM
            elif priority_score >= 35:
                return Priority.LOW
            else:
                return Priority.MINIMAL
                
        except Exception as e:
            logger.error(f"优先级确定失败: {e}")
            return Priority.MEDIUM
    
    def _calculate_confidence(self, data: InnovationData) -> float:
        """计算置信度"""
        try:
            # 基于数据完整性
            data_completeness = self._assess_data_completeness(data)
            
            # 基于历史成功率
            historical_accuracy = data.success_rate
            
            # 基于数据可靠性
            data_reliability = self._assess_data_reliability(data)
            
            # 基于模型稳定性
            model_stability = self._assess_model_stability(data)
            
            confidence = (
                data_completeness * 0.25 +
                historical_accuracy * 0.35 +
                data_reliability * 0.25 +
                model_stability * 0.15
            )
            
            return max(0, min(1, confidence))
            
        except Exception as e:
            logger.error(f"置信度计算失败: {e}")
            return 0.5
    
    def batch_evaluate(self, innovations: List[InnovationData]) -> List[InnovationMetrics]:
        """
        批量评估创新项目
        
        Args:
            innovations: 创新项目列表
            
        Returns:
            List[InnovationMetrics]: 评估结果列表
        """
        logger.info(f"开始批量评估 {len(innovations)} 个创新项目")
        
        results = []
        for innovation in innovations:
            try:
                metrics = self.evaluate_innovation(innovation)
                results.append(metrics)
            except Exception as e:
                logger.error(f"评估项目 {innovation.name} 失败: {e}")
                # 返回默认指标
                default_metrics = InnovationMetrics()
                results.append(default_metrics)
        
        # 按综合得分排序
        results.sort(key=lambda x: x.overall_score, reverse=True)
        
        logger.info("批量评估完成")
        return results
    
    def prioritize_innovations(self, innovations: List[InnovationData]) -> List[Tuple[InnovationData, InnovationMetrics, Priority]]:
        """
        创新项目优先级排序
        
        Args:
            innovations: 创新项目列表
            
        Returns:
            List[Tuple]: (项目数据, 评估指标, 优先级) 的元组列表
        """
        logger.info("开始创新项目优先级排序")
        
        # 批量评估
        metrics_list = self.batch_evaluate(innovations)
        
        # 创建项目-指标-优先级元组
        prioritized_list = []
        for i, (innovation, metrics) in enumerate(zip(innovations, metrics_list)):
            priority = self._determine_detailed_priority(innovation, metrics)
            prioritized_list.append((innovation, metrics, priority))
        
        # 按优先级和得分排序
        prioritized_list.sort(key=lambda x: (x[2].value, -x[1].overall_score))
        
        logger.info("优先级排序完成")
        return prioritized_list
    
    def _determine_detailed_priority(self, innovation: InnovationData, metrics: InnovationMetrics) -> Priority:
        """确定详细优先级"""
        # 综合考虑多个因素
        factors = {
            "score": metrics.overall_score,
            "novelty": metrics.novelty_score,
            "impact": metrics.impact_score,
            "feasibility": metrics.feasibility_score,
            "risk_inverse": 100 - metrics.risk_score,  # 风险越低优先级越高
            "confidence": metrics.confidence_level * 100
        }
        
        # 计算加权优先级分数
        weights = {
            "score": 0.25,
            "novelty": 0.15,
            "impact": 0.2,
            "feasibility": 0.2,
            "risk_inverse": 0.15,
            "confidence": 0.05
        }
        
        priority_score = sum(factors[key] * weights[key] for key in factors)
        
        # 根据分数确定优先级
        if priority_score >= 80:
            return Priority.CRITICAL
        elif priority_score >= 65:
            return Priority.HIGH
        elif priority_score >= 50:
            return Priority.MEDIUM
        elif priority_score >= 35:
            return Priority.LOW
        else:
            return Priority.MINIMAL
    
    def generate_improvement_suggestions(self, innovation: InnovationData, metrics: InnovationMetrics) -> Dict[str, List[str]]:
        """
        生成改进建议
        
        Args:
            innovation: 创新项目数据
            metrics: 评估指标
            
        Returns:
            Dict: 改进建议字典
        """
        logger.info(f"为项目 {innovation.name} 生成改进建议")
        
        suggestions = {
            "创新性提升": [],
            "质量改进": [],
            "可行性增强": [],
            "影响扩大": [],
            "风险降低": [],
            "优先级提升": []
        }
        
        # 创新性改进建议
        if metrics.novelty_score < 70:
            suggestions["创新性提升"].extend([
                "探索跨领域技术融合，增加创新维度",
                "深入研究前沿技术，提升技术独特性",
                "分析市场空白，寻找差异化创新机会",
                "加强原创性思维，避免模仿性创新"
            ])
        
        # 质量改进建议
        if metrics.quality_score < 70:
            suggestions["质量改进"].extend([
                "完善技术方案，提升技术成熟度",
                "加强团队培训，提升专业能力",
                "建立质量管理体系，确保实施质量",
                "增加技术验证环节，降低技术风险"
            ])
        
        # 可行性增强建议
        if metrics.feasibility_score < 70:
            suggestions["可行性增强"].extend([
                "重新评估资源需求，优化资源配置",
                "制定详细的实施计划和时间表",
                "加强与相关方的沟通协调",
                "分阶段实施，降低实施风险"
            ])
        
        # 影响扩大建议
        if metrics.impact_score < 70:
            suggestions["影响扩大"].extend([
                "扩大目标市场范围，增加市场覆盖",
                "加强品牌建设和市场推广",
                "寻找合作伙伴，扩大影响力",
                "关注社会价值创造，提升社会影响"
            ])
        
        # 风险降低建议
        if metrics.risk_score > 60:
            suggestions["风险降低"].extend([
                "建立风险监控和预警机制",
                "制定风险应对预案和备选方案",
                "加强技术验证和测试",
                "分散投资风险，采用多元化策略"
            ])
        
        # 优先级提升建议
        if metrics.priority_level.value > 3:
            suggestions["优先级提升"].extend([
                "重点关注高得分维度的进一步提升",
                "寻找快速见效的改进机会",
                "加强项目管理，确保按时交付",
                "增加资源投入，提升项目重要性"
            ])
        
        logger.info(f"改进建议生成完成，共 {len(suggestions)} 个类别")
        return suggestions
    
    def predict_success_probability(self, innovation: InnovationData) -> Dict[str, float]:
        """
        预测成功概率
        
        Args:
            innovation: 创新项目数据
            
        Returns:
            Dict: 成功概率预测结果
        """
        try:
            logger.info(f"预测项目 {innovation.name} 的成功概率")
            
            # 评估当前状态
            metrics = self.evaluate_innovation(innovation)
            
            # 基于多维度指标预测成功概率
            success_factors = {
                "创新性": metrics.novelty_score / 100,
                "质量": metrics.quality_score / 100,
                "可行性": metrics.feasibility_score / 100,
                "影响": metrics.impact_score / 100,
                "风险控制": (100 - metrics.risk_score) / 100,
                "团队能力": innovation.expertise_level,
                "市场条件": innovation.market_demand,
                "资金支持": innovation.funding_availability
            }
            
            # 计算加权成功概率
            weights = {
                "创新性": 0.15,
                "质量": 0.2,
                "可行性": 0.2,
                "影响": 0.15,
                "风险控制": 0.15,
                "团队能力": 0.1,
                "市场条件": 0.05,
                "资金支持": 0.05
            }
            
            overall_success_probability = sum(
                success_factors[factor] * weights[factor] 
                for factor in success_factors
            )
            
            # 分类别预测
            category_predictions = {
                "技术成功概率": self._predict_technical_success(innovation, metrics),
                "市场成功概率": self._predict_market_success(innovation, metrics),
                "财务成功概率": self._predict_financial_success(innovation, metrics),
                "整体成功概率": overall_success_probability
            }
            
            # 添加置信区间
            confidence_interval = self._calculate_confidence_interval(
                overall_success_probability, metrics.confidence_level
            )
            
            category_predictions["置信区间下限"] = confidence_interval["lower"]
            category_predictions["置信区间上限"] = confidence_interval["upper"]
            
            logger.info(f"成功概率预测完成: {overall_success_probability:.2%}")
            return category_predictions
            
        except Exception as e:
            logger.error(f"成功概率预测失败: {e}")
            return {"整体成功概率": 0.5, "置信区间下限": 0.3, "置信区间上限": 0.7}
    
    def analyze_innovation_trends(self, time_period: int = 12) -> Dict[str, Any]:
        """
        分析创新趋势
        
        Args:
            time_period: 时间周期（月）
            
        Returns:
            Dict: 趋势分析结果
        """
        try:
            logger.info(f"分析最近 {time_period} 个月的创新趋势")
            
            if len(self.evaluation_history) < 10:
                logger.warning("评估历史数据不足，无法进行趋势分析")
                return {"message": "数据不足，无法进行趋势分析"}
            
            # 筛选时间范围内的数据
            cutoff_date = datetime.now() - timedelta(days=time_period * 30)
            recent_evaluations = [
                eval_data for eval_data in self.evaluation_history
                if eval_data["timestamp"] >= cutoff_date
            ]
            
            if len(recent_evaluations) < 5:
                logger.warning("近期评估数据不足")
                return {"message": "近期数据不足"}
            
            # 分析趋势
            trends = {
                "评估数量趋势": self._analyze_evaluation_volume_trend(recent_evaluations),
                "质量趋势": self._analyze_quality_trend(recent_evaluations),
                "创新性趋势": self._analyze_novelty_trend(recent_evaluations),
                "成功率趋势": self._analyze_success_rate_trend(recent_evaluations),
                "风险趋势": self._analyze_risk_trend(recent_evaluations),
                "热门领域": self._analyze_popular_domains(recent_evaluations),
                "平均得分变化": self._analyze_average_score_trend(recent_evaluations)
            }
            
            logger.info("创新趋势分析完成")
            return trends
            
        except Exception as e:
            logger.error(f"创新趋势分析失败: {e}")
            return {"error": str(e)}
    
    def export_evaluation_report(self, innovations: List[InnovationData], 
                               output_file: str = "innovation_evaluation_report.json"):
        """
        导出评估报告
        
        Args:
            innovations: 创新项目列表
            output_file: 输出文件路径
        """
        try:
            logger.info("开始导出评估报告")
            
            # 批量评估
            metrics_list = self.batch_evaluate(innovations)
            
            # 生成报告数据
            report_data = {
                "report_info": {
                    "生成时间": datetime.now().isoformat(),
                    "评估项目数量": len(innovations),
                    "评估器版本": "1.0.0"
                },
                "项目评估结果": []
            }
            
            for innovation, metrics in zip(innovations, metrics_list):
                project_result = {
                    "项目信息": {
                        "项目ID": innovation.project_id,
                        "项目名称": innovation.name,
                        "创新类型": innovation.innovation_type.value,
                        "领域": innovation.domain
                    },
                    "评估指标": {
                        "创新性得分": round(metrics.novelty_score, 2),
                        "质量得分": round(metrics.quality_score, 2),
                        "可行性得分": round(metrics.feasibility_score, 2),
                        "影响得分": round(metrics.impact_score, 2),
                        "风险得分": round(metrics.risk_score, 2),
                        "综合得分": round(metrics.overall_score, 2),
                        "优先级": metrics.priority_level.name,
                        "置信度": round(metrics.confidence_level, 3)
                    },
                    "改进建议": self.generate_improvement_suggestions(innovation, metrics),
                    "成功概率预测": self.predict_success_probability(innovation)
                }
                
                report_data["项目评估结果"].append(project_result)
            
            # 添加统计摘要
            report_data["统计摘要"] = self._generate_statistical_summary(metrics_list)
            
            # 添加趋势分析
            report_data["趋势分析"] = self.analyze_innovation_trends()
            
            # 导出到JSON文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"评估报告已导出到: {output_file}")
            
        except Exception as e:
            logger.error(f"导出评估报告失败: {e}")
            raise
    
    # 辅助方法实现
    def _prepare_features(self, data: InnovationData) -> np.ndarray:
        """准备机器学习特征向量"""
        features = np.array([
            data.technology_readiness,
            data.complexity_level,
            data.resource_requirement,
            data.market_size / 1000,  # 标准化
            data.competition_level,
            data.time_to_market,
            data.team_size,
            data.expertise_level,
            data.experience_years,
            data.success_rate,
            data.market_demand,
            data.funding_availability
        ])
        return features.reshape(1, -1)
    
    # 以下是各种评估辅助方法的简化实现
    def _evaluate_domain_combination(self, data: InnovationData) -> float:
        """评估领域组合创新性"""
        # 简化实现
        return 60.0 + np.random.normal(0, 10)
    
    def _evaluate_tech_fusion(self, data: InnovationData) -> float:
        """评估技术融合程度"""
        return 50.0 + data.complexity_level * 8
    
    def _evaluate_concept_novelty(self, data: InnovationData) -> float:
        """评估概念新颖性"""
        return 55.0 + (9 - data.technology_readiness) * 5
    
    def _evaluate_tech_uniqueness(self, data: InnovationData) -> float:
        """评估技术独特性"""
        return 60.0 - data.competition_level * 10
    
    def _evaluate_application_uniqueness(self, data: InnovationData) -> float:
        """评估应用独特性"""
        return 65.0 + np.random.normal(0, 8)
    
    def _evaluate_requirement_clarity(self, data: InnovationData) -> float:
        """评估需求明确性"""
        return 70.0 + data.expertise_level * 20
    
    def _evaluate_solution_completeness(self, data: InnovationData) -> float:
        """评估解决方案完整性"""
        return data.technology_readiness * 10 + 10
    
    def _evaluate_plan_completeness(self, data: InnovationData) -> float:
        """评估计划完整性"""
        return 75.0 - data.complexity_level * 5
    
    def _evaluate_goal_consistency(self, data: InnovationData) -> float:
        """评估目标一致性"""
        return 80.0 + np.random.normal(0, 5)
    
    def _evaluate_tech_consistency(self, data: InnovationData) -> float:
        """评估技术一致性"""
        return data.technology_readiness * 8 + 20
    
    def _evaluate_resource_consistency(self, data: InnovationData) -> float:
        """评估资源一致性"""
        return data.funding_availability * 100
    
    def _estimate_roi(self, data: InnovationData) -> float:
        """估算投资回报率"""
        base_roi = data.market_size / 10  # 假设每100万市场对应10% ROI
        return min(100, base_roi + np.random.normal(0, 10))
    
    def _calculate_cost_benefit_ratio(self, data: InnovationData) -> float:
        """计算成本效益比"""
        benefit_estimate = data.market_size / 5
        cost_estimate = data.resource_requirement * 100
        ratio = benefit_estimate / max(cost_estimate, 1)
        return min(100, ratio * 10)
    
    def _evaluate_team_size_adequacy(self, data: InnovationData) -> float:
        """评估团队规模充足性"""
        optimal_size = 5
        size_diff = abs(data.team_size - optimal_size)
        return max(0, 100 - size_diff * 15)
    
    def _evaluate_organizational_support(self, data: InnovationData) -> float:
        """评估组织支持度"""
        return 70.0 + data.funding_availability * 25
    
    def _evaluate_supply_chain_readiness(self, data: InnovationData) -> float:
        """评估供应链准备度"""
        return 65.0 + np.random.normal(0, 10)
    
    def _evaluate_process_maturity(self, data: InnovationData) -> float:
        """评估流程成熟度"""
        return data.technology_readiness * 8 + 20
    
    def _evaluate_market_timing(self, data: InnovationData) -> float:
        """评估市场时机"""
        return data.market_demand * 100
    
    def _evaluate_social_need_match(self, data: InnovationData) -> float:
        """评估社会需求匹配度"""
        return 60.0 + data.market_demand * 30
    
    def _evaluate_social_acceptance(self, data: InnovationData) -> float:
        """评估社会接受度"""
        return 70.0 + np.random.normal(0, 15)
    
    def _evaluate_environmental_impact(self, data: InnovationData) -> float:
        """评估环境影响"""
        return 75.0 + np.random.normal(0, 10)
    
    def _evaluate_social_value_creation(self, data: InnovationData) -> float:
        """评估社会价值创造"""
        return data.market_size / 20 + 50
    
    def _evaluate_tech_advancement(self, data: InnovationData) -> float:
        """评估技术先进性"""
        return (9 - data.technology_readiness) * 10 + 20
    
    def _evaluate_tech_diffusion_potential(self, data: InnovationData) -> float:
        """评估技术扩散潜力"""
        return data.complexity_level * 15 + 25
    
    def _evaluate_tech_ecosystem_impact(self, data: InnovationData) -> float:
        """评估技术生态影响"""
        return 65.0 + np.random.normal(0, 12)
    
    def _evaluate_tech_standardization_potential(self, data: InnovationData) -> float:
        """评估技术标准化潜力"""
        return data.technology_readiness * 8 + 30
    
    def _estimate_economic_value_creation(self, data: InnovationData) -> float:
        """估算经济价值创造"""
        return min(100, data.market_size / 3 + np.random.normal(0, 10))
    
    def _estimate_job_creation_potential(self, data: InnovationData) -> float:
        """估算就业创造潜力"""
        return data.team_size * 8 + data.market_size / 50
    
    def _evaluate_industry_upgrade_contribution(self, data: InnovationData) -> float:
        """评估产业升级贡献"""
        return 60.0 + data.complexity_level * 8
    
    def _evaluate_national_competitiveness_boost(self, data: InnovationData) -> float:
        """评估国家竞争力提升"""
        return 55.0 + data.technology_readiness * 6
    
    def _estimate_market_growth_rate(self, data: InnovationData) -> float:
        """估算市场增长率"""
        return 5.0 + data.market_demand * 15  # 百分比
    
    def _estimate_market_penetration_potential(self, data: InnovationData) -> float:
        """估算市场渗透潜力"""
        return 70.0 - data.competition_level * 12
    
    def _assess_competitive_advantage(self, data: InnovationData) -> float:
        """评估竞争优势"""
        return 65.0 + data.expertise_level * 25
    
    def _assess_tech_dependency_risk(self, data: InnovationData) -> float:
        """评估技术依赖风险"""
        return data.complexity_level * 12 + 20
    
    def _assess_market_change_risk(self, data: InnovationData) -> float:
        """评估市场变化风险"""
        return 40.0 + np.random.normal(0, 15)
    
    def _assess_roi_uncertainty(self, data: InnovationData) -> float:
        """评估投资回报不确定性"""
        return data.complexity_level * 15 + 25
    
    def _assess_cost_overrun_risk(self, data: InnovationData) -> float:
        """评估成本超支风险"""
        return data.resource_requirement * 80 + 10
    
    def _assess_cash_flow_risk(self, data: InnovationData) -> float:
        """评估现金流风险"""
        return max(0, 100 - data.funding_availability * 100)
    
    def _assess_supply_chain_risk(self, data: InnovationData) -> float:
        """评估供应链风险"""
        return 35.0 + np.random.normal(0, 12)
    
    def _assess_quality_control_risk(self, data: InnovationData) -> float:
        """评估质量控制风险"""
        return max(0, 100 - data.expertise_level * 100)
    
    def _assess_compliance_complexity(self, data: InnovationData) -> float:
        """评估合规复杂度"""
        return data.complexity_level * 18 + 15
    
    def _assess_policy_change_risk(self, data: InnovationData) -> float:
        """评估政策变化风险"""
        return 30.0 + np.random.normal(0, 10)
    
    def _assess_international_regulatory_risk(self, data: InnovationData) -> float:
        """评估国际监管风险"""
        return 25.0 + np.random.normal(0, 8)
    
    def _assess_data_completeness(self, data: InnovationData) -> float:
        """评估数据完整性"""
        completeness_factors = [
            1 if data.description and len(data.description.strip()) > 0 else 0,
            data.technology_readiness,
            data.market_size,
            data.team_size,
            data.expertise_level
        ]
        return sum(1 for factor in completeness_factors if factor > 0) / len(completeness_factors)
    
    def _assess_data_reliability(self, data: InnovationData) -> float:
        """评估数据可靠性"""
        return data.success_rate * 100
    
    def _assess_model_stability(self, data: InnovationData) -> float:
        """评估模型稳定性"""
        return 75.0 + np.random.normal(0, 10)
    
    def _predict_technical_success(self, innovation: InnovationData, metrics: InnovationMetrics) -> float:
        """预测技术成功概率"""
        tech_factors = {
            "技术成熟度": innovation.technology_readiness / 9,
            "团队专业能力": innovation.expertise_level,
            "技术复杂度适配": max(0, 1 - abs(innovation.complexity_level - 3) / 4),
            "技术质量": metrics.quality_score / 100
        }
        return sum(tech_factors.values()) / len(tech_factors)
    
    def _predict_market_success(self, innovation: InnovationData, metrics: InnovationMetrics) -> float:
        """预测市场成功概率"""
        market_factors = {
            "市场需求": innovation.market_demand,
            "市场规模": min(1, innovation.market_size / 1000),
            "竞争环境": max(0, 1 - innovation.competition_level / 5),
            "市场影响": metrics.impact_score / 100
        }
        return sum(market_factors.values()) / len(market_factors)
    
    def _predict_financial_success(self, innovation: InnovationData, metrics: InnovationMetrics) -> float:
        """预测财务成功概率"""
        financial_factors = {
            "资金可获得性": innovation.funding_availability,
            "投资回报预期": min(1, innovation.market_size / 500),
            "成本控制": max(0, 1 - innovation.resource_requirement),
            "风险控制": (100 - metrics.risk_score) / 100
        }
        return sum(financial_factors.values()) / len(financial_factors)
    
    def _calculate_confidence_interval(self, probability: float, confidence: float) -> Dict[str, float]:
        """计算置信区间"""
        margin = (1 - confidence) * 0.2  # 简化的置信区间计算
        return {
            "lower": max(0, probability - margin),
            "upper": min(1, probability + margin)
        }
    
    def _analyze_evaluation_volume_trend(self, evaluations: List) -> Dict[str, Any]:
        """分析评估数量趋势"""
        return {"趋势": "稳定", "月均评估数": len(evaluations) // 12 + 1}
    
    def _analyze_quality_trend(self, evaluations: List) -> Dict[str, Any]:
        """分析质量趋势"""
        avg_quality = np.mean([eval_data["metrics"].quality_score for eval_data in evaluations])
        return {"平均质量得分": round(avg_quality, 2), "趋势": "稳定"}
    
    def _analyze_novelty_trend(self, evaluations: List) -> Dict[str, Any]:
        """分析创新性趋势"""
        avg_novelty = np.mean([eval_data["metrics"].novelty_score for eval_data in evaluations])
        return {"平均创新性得分": round(avg_novelty, 2), "趋势": "稳定"}
    
    def _analyze_success_rate_trend(self, evaluations: List) -> Dict[str, Any]:
        """分析成功率趋势"""
        avg_success = np.mean([eval_data["metrics"].overall_score for eval_data in evaluations])
        return {"平均成功率": round(avg_success / 100, 3), "趋势": "稳定"}
    
    def _analyze_risk_trend(self, evaluations: List) -> Dict[str, Any]:
        """分析风险趋势"""
        avg_risk = np.mean([eval_data["metrics"].risk_score for eval_data in evaluations])
        return {"平均风险得分": round(avg_risk, 2), "趋势": "稳定"}
    
    def _analyze_popular_domains(self, evaluations: List) -> List[str]:
        """分析热门领域"""
        # 简化实现，返回一些示例领域
        return ["人工智能", "新能源", "生物技术", "区块链", "物联网"]
    
    def _analyze_average_score_trend(self, evaluations: List) -> Dict[str, float]:
        """分析平均得分趋势"""
        avg_score = np.mean([eval_data["metrics"].overall_score for eval_data in evaluations])
        return {"当前平均得分": round(avg_score, 2)}
    
    def _generate_statistical_summary(self, metrics_list: List[InnovationMetrics]) -> Dict[str, Any]:
        """生成统计摘要"""
        if not metrics_list:
            return {}
        
        scores = [metrics.overall_score for metrics in metrics_list]
        novelties = [metrics.novelty_score for metrics in metrics_list]
        qualities = [metrics.quality_score for metrics in metrics_list]
        impacts = [metrics.impact_score for metrics in metrics_list]
        risks = [metrics.risk_score for metrics in metrics_list]
        
        return {
            "综合得分统计": {
                "平均值": round(np.mean(scores), 2),
                "中位数": round(np.median(scores), 2),
                "标准差": round(np.std(scores), 2),
                "最高分": round(max(scores), 2),
                "最低分": round(min(scores), 2)
            },
            "创新性得分统计": {
                "平均值": round(np.mean(novelties), 2),
                "中位数": round(np.median(novelties), 2)
            },
            "质量得分统计": {
                "平均值": round(np.mean(qualities), 2),
                "中位数": round(np.median(qualities), 2)
            },
            "影响得分统计": {
                "平均值": round(np.mean(impacts), 2),
                "中位数": round(np.median(impacts), 2)
            },
            "风险得分统计": {
                "平均值": round(np.mean(risks), 2),
                "中位数": round(np.median(risks), 2)
            },
            "优先级分布": self._calculate_priority_distribution(metrics_list)
        }
    
    def _calculate_priority_distribution(self, metrics_list: List[InnovationMetrics]) -> Dict[str, int]:
        """计算优先级分布"""
        distribution = {}
        for priority in Priority:
            distribution[priority.name] = sum(
                1 for metrics in metrics_list if metrics.priority_level == priority
            )
        return distribution


# 示例使用和测试代码
def create_sample_innovations() -> List[InnovationData]:
    """创建示例创新项目数据"""
    return [
        InnovationData(
            project_id="INN001",
            name="智能医疗诊断系统",
            description="基于AI的医疗影像诊断系统",
            innovation_type=InnovationType.BREAKTHROUGH,
            category="医疗科技",
            subcategory="诊断设备",
            domain="医疗健康",
            technology_readiness=6,
            complexity_level=4,
            resource_requirement=0.8,
            market_size=500.0,
            competition_level=3,
            time_to_market=18,
            team_size=12,
            expertise_level=0.85,
            experience_years=8.5,
            success_rate=0.75,
            market_demand=0.9,
            funding_availability=0.8,
            regulatory_support=0.7
        ),
        InnovationData(
            project_id="INN002",
            name="新能源汽车电池技术",
            description="高能量密度固态电池技术",
            innovation_type=InnovationType.RADICAL,
            category="新能源",
            subcategory="电池技术",
            domain="能源环保",
            technology_readiness=5,
            complexity_level=5,
            resource_requirement=0.9,
            market_size=1200.0,
            competition_level=4,
            time_to_market=24,
            team_size=20,
            expertise_level=0.9,
            experience_years=12.0,
            success_rate=0.6,
            market_demand=0.95,
            funding_availability=0.9,
            regulatory_support=0.8
        ),
        InnovationData(
            project_id="INN003",
            name="智能农业管理系统",
            description="基于物联网的精准农业管理平台",
            innovation_type=InnovationType.INCREMENTAL,
            category="农业科技",
            subcategory="智能农业",
            domain="农业食品",
            technology_readiness=7,
            complexity_level=3,
            resource_requirement=0.6,
            market_size=300.0,
            competition_level=2,
            time_to_market=12,
            team_size=8,
            expertise_level=0.75,
            experience_years=6.0,
            success_rate=0.8,
            market_demand=0.8,
            funding_availability=0.7,
            regulatory_support=0.6
        )
    ]


def main():
    """主函数 - 演示创新评估器的使用"""
    print("=== 创新评估器演示 ===")
    
    # 创建评估器
    evaluator = InnovationEvaluator()
    
    # 创建示例数据
    innovations = create_sample_innovations()
    
    print(f"\n评估 {len(innovations)} 个创新项目:")
    
    # 逐一评估
    for innovation in innovations:
        print(f"\n--- 评估项目: {innovation.name} ---")
        
        # 评估指标
        metrics = evaluator.evaluate_innovation(innovation)
        print(f"综合得分: {metrics.overall_score:.2f}")
        print(f"创新性得分: {metrics.novelty_score:.2f}")
        print(f"质量得分: {metrics.quality_score:.2f}")
        print(f"可行性得分: {metrics.feasibility_score:.2f}")
        print(f"影响得分: {metrics.impact_score:.2f}")
        print(f"风险得分: {metrics.risk_score:.2f}")
        print(f"优先级: {metrics.priority_level.name}")
        print(f"置信度: {metrics.confidence_level:.3f}")
        
        # 成功概率预测
        success_probs = evaluator.predict_success_probability(innovation)
        print(f"整体成功概率: {success_probs['整体成功概率']:.2%}")
        
        # 改进建议
        suggestions = evaluator.generate_improvement_suggestions(innovation, metrics)
        print("主要改进建议:")
        for category, suggestion_list in suggestions.items():
            if suggestion_list:
                print(f"  {category}: {suggestion_list[0]}")
    
    # 批量评估和优先级排序
    print(f"\n--- 批量评估和优先级排序 ---")
    prioritized_list = evaluator.prioritize_innovations(innovations)
    
    for i, (innovation, metrics, priority) in enumerate(prioritized_list, 1):
        print(f"{i}. {innovation.name} (优先级: {priority.name}, 得分: {metrics.overall_score:.2f})")
    
    # 趋势分析
    print(f"\n--- 创新趋势分析 ---")
    trends = evaluator.analyze_innovation_trends()
    for trend_name, trend_data in trends.items():
        print(f"{trend_name}: {trend_data}")
    
    # 导出报告
    print(f"\n--- 导出评估报告 ---")
    evaluator.export_evaluation_report(innovations, "innovation_evaluation_report.json")
    print("评估报告已导出到 innovation_evaluation_report.json")
    
    print(f"\n=== 演示完成 ===")


if __name__ == "__main__":
    main()