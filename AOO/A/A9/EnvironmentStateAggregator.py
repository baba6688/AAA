#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A9环境状态聚合器
实现多源数据融合、环境状态评估、市场环境分类、趋势分析、状态预测和综合指标生成
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketEnvironment(Enum):
    """市场环境枚举"""
    BULL_MARKET = "牛市"
    BEAR_MARKET = "熊市"
    SIDEWAYS = "震荡市"
    VOLATILE = "高波动市"
    STABLE = "稳定市"
    UNKNOWN = "未知"


class TrendDirection(Enum):
    """趋势方向枚举"""
    STRONG_UP = "强势上涨"
    WEAK_UP = "弱势上涨"
    SIDEWAYS = "横盘震荡"
    WEAK_DOWN = "弱势下跌"
    STRONG_DOWN = "强势下跌"
    UNKNOWN = "未知"


@dataclass
class DataSource:
    """数据源配置"""
    name: str
    weight: float
    data_type: str
    last_update: datetime = field(default_factory=datetime.now)
    reliability: float = 1.0  # 数据可靠性评分 0-1
    update_frequency: int = 60  # 更新频率（秒）


@dataclass
class EnvironmentMetrics:
    """环境指标数据"""
    timestamp: datetime
    volatility: float = 0.0  # 波动率
    momentum: float = 0.0  # 动量
    trend_strength: float = 0.0  # 趋势强度
    liquidity: float = 0.0  # 流动性
    sentiment: float = 0.0  # 情绪指标
    correlation: float = 0.0  # 相关性
    risk_level: float = 0.0  # 风险水平
    market_cap: float = 0.0  # 市值
    volume_ratio: float = 0.0  # 成交量比率


@dataclass
class EnvironmentState:
    """环境状态"""
    timestamp: datetime
    environment_type: MarketEnvironment
    trend_direction: TrendDirection
    confidence: float  # 状态置信度 0-1
    score: float  # 综合评分 -1到1
    metrics: EnvironmentMetrics
    factors: Dict[str, float] = field(default_factory=dict)  # 影响因素
    prediction: Dict[str, Any] = field(default_factory=dict)  # 预测数据


class DataFusionEngine:
    """数据融合引擎"""
    
    def __init__(self, fusion_method: str = "weighted_average"):
        self.fusion_method = fusion_method
        self.weights_cache = {}
        
    def fuse_data(self, data_sources: List[DataSource], 
                  raw_data: Dict[str, Any]) -> Dict[str, float]:
        """融合多源数据"""
        if not raw_data:
            return {}
            
        fused_data = {}
        
        for source in data_sources:
            if source.name in raw_data:
                # 应用权重和可靠性
                effective_weight = source.weight * source.reliability
                
                for key, value in raw_data[source.name].items():
                    if key not in fused_data:
                        fused_data[key] = []
                    fused_data[key].append((value, effective_weight))
        
        # 应用融合方法
        for key, value_weight_pairs in fused_data.items():
            if self.fusion_method == "weighted_average":
                values, weights = zip(*value_weight_pairs)
                fused_data[key] = np.average(values, weights=weights)
            elif self.fusion_method == "median":
                values = [v[0] for v in value_weight_pairs]
                fused_data[key] = np.median(values)
            elif self.fusion_method == "robust_mean":
                values = [v[0] for v in value_weight_pairs]
                weights = [v[1] for v in value_weight_pairs]
                # 去除异常值后计算加权平均
                q25, q75 = np.percentile(values, [25, 75])
                filtered_values = [v for v in values if q25 <= v <= q75]
                filtered_weights = [weights[i] for i, v in enumerate(values) 
                                  if q25 <= v <= q75]
                if filtered_values:
                    fused_data[key] = np.average(filtered_values, weights=filtered_weights)
                else:
                    fused_data[key] = np.mean(values)
        
        return fused_data


class EnvironmentClassifier:
    """环境分类器"""
    
    def __init__(self):
        self.classification_rules = {
            MarketEnvironment.BULL_MARKET: {
                'momentum_threshold': 0.3,
                'trend_threshold': 0.4,
                'volatility_max': 0.8
            },
            MarketEnvironment.BEAR_MARKET: {
                'momentum_threshold': -0.3,
                'trend_threshold': -0.4,
                'volatility_max': 0.8
            },
            MarketEnvironment.SIDEWAYS: {
                'momentum_range': (-0.2, 0.2),
                'trend_range': (-0.3, 0.3),
                'volatility_max': 0.6
            },
            MarketEnvironment.VOLATILE: {
                'volatility_threshold': 0.7
            },
            MarketEnvironment.STABLE: {
                'volatility_max': 0.3,
                'trend_range': (-0.2, 0.2)
            }
        }
    
    def classify_environment(self, metrics: EnvironmentMetrics) -> MarketEnvironment:
        """分类市场环境"""
        scores = {}
        
        # 牛市判断
        if (metrics.momentum > self.classification_rules[MarketEnvironment.BULL_MARKET]['momentum_threshold'] and
            metrics.trend_strength > self.classification_rules[MarketEnvironment.BULL_MARKET]['trend_threshold'] and
            metrics.volatility <= self.classification_rules[MarketEnvironment.BULL_MARKET]['volatility_max']):
            scores[MarketEnvironment.BULL_MARKET] = 0.8
        
        # 熊市判断
        if (metrics.momentum < self.classification_rules[MarketEnvironment.BEAR_MARKET]['momentum_threshold'] and
            metrics.trend_strength < self.classification_rules[MarketEnvironment.BEAR_MARKET]['trend_threshold'] and
            metrics.volatility <= self.classification_rules[MarketEnvironment.BEAR_MARKET]['volatility_max']):
            scores[MarketEnvironment.BEAR_MARKET] = 0.8
        
        # 震荡市判断
        rules = self.classification_rules[MarketEnvironment.SIDEWAYS]
        if (rules['momentum_range'][0] <= metrics.momentum <= rules['momentum_range'][1] and
            rules['trend_range'][0] <= metrics.trend_strength <= rules['trend_range'][1] and
            metrics.volatility <= rules['volatility_max']):
            scores[MarketEnvironment.SIDEWAYS] = 0.7
        
        # 高波动市判断
        if metrics.volatility > self.classification_rules[MarketEnvironment.VOLATILE]['volatility_threshold']:
            scores[MarketEnvironment.VOLATILE] = metrics.volatility
        
        # 稳定市判断
        rules = self.classification_rules[MarketEnvironment.STABLE]
        if (metrics.volatility <= rules['volatility_max'] and
            rules['trend_range'][0] <= metrics.trend_strength <= rules['trend_range'][1]):
            scores[MarketEnvironment.STABLE] = 0.6
        
        # 返回得分最高的环境类型
        if scores:
            return max(scores.keys(), key=lambda k: scores[k])
        else:
            return MarketEnvironment.UNKNOWN
    
    def classify_trend(self, metrics: EnvironmentMetrics) -> TrendDirection:
        """分类趋势方向"""
        momentum = metrics.momentum
        trend_strength = metrics.trend_strength
        
        if momentum > 0.5 and trend_strength > 0.4:
            return TrendDirection.STRONG_UP
        elif momentum > 0.1 and trend_strength > 0.2:
            return TrendDirection.WEAK_UP
        elif -0.1 <= momentum <= 0.1 and -0.2 <= trend_strength <= 0.2:
            return TrendDirection.SIDEWAYS
        elif momentum < -0.1 and trend_strength < -0.2:
            return TrendDirection.WEAK_DOWN
        elif momentum < -0.5 and trend_strength < -0.4:
            return TrendDirection.STRONG_DOWN
        else:
            return TrendDirection.UNKNOWN


class TrendAnalyzer:
    """趋势分析器"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.score_history = deque(maxlen=window_size)
    
    def analyze_trend(self, current_metrics: EnvironmentMetrics, current_score: float = 0.0) -> Dict[str, Any]:
        """分析趋势变化"""
        self.metrics_history.append(current_metrics)
        self.score_history.append(current_score)
        
        if len(self.score_history) < 3:
            return {'trend': 'insufficient_data', 'strength': 0.0, 'acceleration': 0.0, 'stability': 0.0}
        
        # 计算趋势强度
        recent_scores = list(self.score_history)[-5:]
        trend_strength = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
        # 计算加速度
        if len(self.score_history) >= 3:
            recent_trends = []
            for i in range(2, len(recent_scores)):
                trend = recent_scores[i] - recent_scores[i-1]
                recent_trends.append(trend)
            
            if len(recent_trends) >= 2:
                acceleration = recent_trends[-1] - recent_trends[-2]
            else:
                acceleration = 0.0
        else:
            acceleration = 0.0
        
        # 判断趋势状态
        if trend_strength > 0.1:
            trend_status = "上升趋势"
        elif trend_strength < -0.1:
            trend_status = "下降趋势"
        else:
            trend_status = "横盘趋势"
        
        return {
            'trend': trend_status,
            'strength': trend_strength,
            'acceleration': acceleration,
            'stability': 1.0 - np.std(recent_scores) if len(recent_scores) > 1 else 0.0
        }


class EnvironmentPredictor:
    """环境状态预测器"""
    
    def __init__(self, prediction_horizon: int = 5):
        self.prediction_horizon = prediction_horizon
        self.model_cache = {}
    
    def predict_environment(self, history: List[EnvironmentState]) -> Dict[str, Any]:
        """预测环境状态"""
        if len(history) < 10:
            return {'prediction': 'insufficient_data', 'confidence': 0.0, 'horizon': self.prediction_horizon}
        
        # 提取特征
        features = self._extract_features(history)
        
        # 简单预测模型（基于历史趋势）
        recent_states = history[-10:]
        environment_counts = {}
        for state in recent_states:
            env_type = state.environment_type
            environment_counts[env_type] = environment_counts.get(env_type, 0) + 1
        
        # 预测最可能的环境
        predicted_environment = max(environment_counts.keys(), 
                                  key=lambda k: environment_counts[k])
        
        # 计算预测置信度
        max_count = environment_counts[predicted_environment]
        confidence = max_count / len(recent_states)
        
        # 趋势预测
        recent_scores = [state.score for state in recent_states[-5:]]
        if len(recent_scores) >= 2:
            trend_slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
            predicted_score_change = trend_slope * self.prediction_horizon
        else:
            predicted_score_change = 0.0
        
        return {
            'predicted_environment': predicted_environment,
            'confidence': confidence,
            'horizon': self.prediction_horizon,
            'predicted_score_change': predicted_score_change,
            'alternative_environments': dict(sorted(environment_counts.items(), 
                                                  key=lambda x: x[1], reverse=True)[:3])
        }
    
    def _extract_features(self, history: List[EnvironmentState]) -> np.ndarray:
        """提取预测特征"""
        if len(history) < 5:
            return np.array([])
        
        recent_states = history[-5:]
        features = []
        
        for state in recent_states:
            # 环境类型编码
            env_encoding = self._encode_environment(state.environment_type)
            features.extend(env_encoding)
            
            # 指标特征
            features.extend([
                state.metrics.volatility,
                state.metrics.momentum,
                state.metrics.trend_strength,
                state.score
            ])
        
        return np.array(features)
    
    def _encode_environment(self, environment: MarketEnvironment) -> List[float]:
        """环境类型编码"""
        encoding = [0.0] * len(MarketEnvironment)
        encoding[environment.value] = 1.0
        return encoding


class StrategyImpactAnalyzer:
    """策略影响分析器"""
    
    def __init__(self):
        self.strategy_profiles = {
            'trend_following': {
                'preferred_environment': [MarketEnvironment.BULL_MARKET, MarketEnvironment.BEAR_MARKET],
                'risk_tolerance': 0.7,
                'performance_weights': {
                    'volatility': -0.3,
                    'trend_strength': 0.8,
                    'momentum': 0.6
                }
            },
            'mean_reversion': {
                'preferred_environment': [MarketEnvironment.SIDEWAYS, MarketEnvironment.STABLE],
                'risk_tolerance': 0.4,
                'performance_weights': {
                    'volatility': -0.5,
                    'trend_strength': -0.3,
                    'momentum': -0.4
                }
            },
            'momentum': {
                'preferred_environment': [MarketEnvironment.BULL_MARKET, MarketEnvironment.VOLATILE],
                'risk_tolerance': 0.8,
                'performance_weights': {
                    'volatility': 0.2,
                    'trend_strength': 0.6,
                    'momentum': 0.9
                }
            },
            'arbitrage': {
                'preferred_environment': [MarketEnvironment.STABLE, MarketEnvironment.SIDEWAYS],
                'risk_tolerance': 0.3,
                'performance_weights': {
                    'volatility': -0.7,
                    'trend_strength': -0.2,
                    'liquidity': 0.8
                }
            }
        }
    
    def analyze_impact(self, environment_state: EnvironmentState, 
                      strategy_name: str) -> Dict[str, Any]:
        """分析环境对策略的影响"""
        if strategy_name not in self.strategy_profiles:
            return {'error': f'Unknown strategy: {strategy_name}'}
        
        profile = self.strategy_profiles[strategy_name]
        metrics = environment_state.metrics
        
        # 计算适配度
        environment_fit = 1.0 if environment_state.environment_type in profile['preferred_environment'] else 0.3
        
        # 计算性能预期
        performance_score = 0.0
        weights = profile['performance_weights']
        
        performance_score += weights.get('volatility', 0) * metrics.volatility
        performance_score += weights.get('trend_strength', 0) * metrics.trend_strength
        performance_score += weights.get('momentum', 0) * metrics.momentum
        performance_score += weights.get('liquidity', 0) * metrics.liquidity
        
        # 风险调整
        risk_adjustment = 1.0 - (metrics.risk_level * (1.0 - profile['risk_tolerance']))
        
        # 综合评分
        expected_performance = performance_score * environment_fit * risk_adjustment
        
        # 建议
        recommendations = self._generate_recommendations(
            environment_state, strategy_name, expected_performance
        )
        
        return {
            'strategy': strategy_name,
            'environment_fit': environment_fit,
            'expected_performance': expected_performance,
            'risk_level': metrics.risk_level,
            'recommendations': recommendations,
            'suitable': expected_performance > 0.3
        }
    
    def _generate_recommendations(self, environment_state: EnvironmentState, 
                                strategy_name: str, performance: float) -> List[str]:
        """生成策略建议"""
        recommendations = []
        
        if performance > 0.7:
            recommendations.append(f"{strategy_name}策略在当前环境下表现优异，建议积极执行")
        elif performance > 0.3:
            recommendations.append(f"{strategy_name}策略在当前环境下表现良好，可以适度执行")
        elif performance > 0.0:
            recommendations.append(f"{strategy_name}策略在当前环境下表现一般，建议谨慎执行")
        else:
            recommendations.append(f"{strategy_name}策略在当前环境下表现较差，建议暂停或调整")
        
        # 基于环境类型的特殊建议
        if environment_state.environment_type == MarketEnvironment.VOLATILE:
            recommendations.append("高波动环境下，建议增加风险控制措施")
        elif environment_state.environment_type == MarketEnvironment.STABLE:
            recommendations.append("稳定环境下，适合执行低风险策略")
        
        return recommendations


class EnvironmentStateAggregator:
    """环境状态聚合器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.data_sources: List[DataSource] = []
        self.fusion_engine = DataFusionEngine()
        self.classifier = EnvironmentClassifier()
        self.trend_analyzer = TrendAnalyzer()
        self.predictor = EnvironmentPredictor()
        self.impact_analyzer = StrategyImpactAnalyzer()
        
        # 状态存储
        self.current_state: Optional[EnvironmentState] = None
        self.state_history: deque = deque(maxlen=1000)
        self.metrics_history: deque = deque(maxlen=100)
        
        # 初始化默认数据源
        self._initialize_default_sources()
        
        logger.info("环境状态聚合器初始化完成")
    
    def _initialize_default_sources(self):
        """初始化默认数据源"""
        default_sources = [
            DataSource("price_data", 0.3, "market_data", reliability=0.9),
            DataSource("volume_data", 0.2, "market_data", reliability=0.8),
            DataSource("volatility_data", 0.25, "technical_data", reliability=0.8),
            DataSource("sentiment_data", 0.15, "sentiment_data", reliability=0.6),
            DataSource("macro_data", 0.1, "fundamental_data", reliability=0.7)
        ]
        self.data_sources.extend(default_sources)
    
    def add_data_source(self, source: DataSource):
        """添加数据源"""
        self.data_sources.append(source)
        logger.info(f"添加数据源: {source.name}")
    
    def update_data_sources(self, raw_data: Dict[str, Any]):
        """更新数据源数据"""
        # 更新数据源时间戳
        for source in self.data_sources:
            if source.name in raw_data:
                source.last_update = datetime.now()
        
        # 融合数据
        fused_data = self.fusion_engine.fuse_data(self.data_sources, raw_data)
        
        # 生成环境指标
        metrics = self._generate_metrics(fused_data)
        
        # 更新状态
        self._update_environment_state(metrics)
        
        return self.current_state
    
    def _generate_metrics(self, fused_data: Dict[str, float]) -> EnvironmentMetrics:
        """生成环境指标"""
        current_time = datetime.now()
        
        # 从融合数据中提取指标
        volatility = fused_data.get('volatility', 0.0)
        momentum = fused_data.get('momentum', 0.0)
        trend_strength = fused_data.get('trend_strength', 0.0)
        liquidity = fused_data.get('liquidity', 0.5)
        sentiment = fused_data.get('sentiment', 0.0)
        correlation = fused_data.get('correlation', 0.0)
        risk_level = fused_data.get('risk_level', 0.5)
        market_cap = fused_data.get('market_cap', 0.0)
        volume_ratio = fused_data.get('volume_ratio', 1.0)
        
        metrics = EnvironmentMetrics(
            timestamp=current_time,
            volatility=volatility,
            momentum=momentum,
            trend_strength=trend_strength,
            liquidity=liquidity,
            sentiment=sentiment,
            correlation=correlation,
            risk_level=risk_level,
            market_cap=market_cap,
            volume_ratio=volume_ratio
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _update_environment_state(self, metrics: EnvironmentMetrics):
        """更新环境状态"""
        # 分类环境和趋势
        environment_type = self.classifier.classify_environment(metrics)
        trend_direction = self.classifier.classify_trend(metrics)
        
        # 计算综合评分
        score = self._calculate_composite_score(metrics)
        
        # 计算置信度
        confidence = self._calculate_confidence(metrics)
        
        # 趋势分析
        trend_analysis = self.trend_analyzer.analyze_trend(metrics, score)
        
        # 环境预测
        prediction = self.predictor.predict_environment(list(self.state_history))
        
        # 创建环境状态
        state = EnvironmentState(
            timestamp=datetime.now(),
            environment_type=environment_type,
            trend_direction=trend_direction,
            confidence=confidence,
            score=score,
            metrics=metrics,
            factors={
                'trend_strength': trend_analysis['strength'],
                'stability': trend_analysis['stability'],
                'acceleration': trend_analysis['acceleration']
            },
            prediction=prediction
        )
        
        self.current_state = state
        self.state_history.append(state)
        
        logger.info(f"环境状态更新: {environment_type.value}, 评分: {score:.3f}")
    
    def _calculate_composite_score(self, metrics: EnvironmentMetrics) -> float:
        """计算综合评分"""
        # 权重配置
        weights = {
            'momentum': 0.25,
            'trend_strength': 0.25,
            'volatility': -0.15,  # 负权重，波动率越低越好
            'liquidity': 0.15,
            'sentiment': 0.1,
            'risk_level': -0.1   # 负权重，风险越低越好
        }
        
        score = 0.0
        for factor, weight in weights.items():
            value = getattr(metrics, factor, 0.0)
            score += weight * value
        
        # 标准化到[-1, 1]范围
        return np.tanh(score)
    
    def _calculate_confidence(self, metrics: EnvironmentMetrics) -> float:
        """计算置信度"""
        # 基于数据完整性和一致性计算置信度
        confidence_factors = []
        
        # 数据新鲜度
        if self.metrics_history:
            last_metrics = self.metrics_history[-1]
            time_diff = (datetime.now() - last_metrics.timestamp).total_seconds()
            freshness_score = max(0.0, 1.0 - time_diff / 300)  # 5分钟内的数据认为新鲜
            confidence_factors.append(freshness_score)
        
        # 指标一致性（各指标值的合理性）
        consistency_score = 1.0
        if abs(metrics.volatility) > 1.0 or abs(metrics.momentum) > 1.0:
            consistency_score = 0.5
        confidence_factors.append(consistency_score)
        
        # 历史数据支持度
        history_support = min(1.0, len(self.metrics_history) / 10.0)
        confidence_factors.append(history_support)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def get_current_state(self) -> Optional[EnvironmentState]:
        """获取当前环境状态"""
        return self.current_state
    
    def get_state_history(self, days: int = 30) -> List[EnvironmentState]:
        """获取历史状态"""
        cutoff_time = datetime.now() - timedelta(days=days)
        return [state for state in self.state_history if state.timestamp >= cutoff_time]
    
    def analyze_strategy_impact(self, strategy_name: str) -> Dict[str, Any]:
        """分析策略影响"""
        if not self.current_state:
            return {'error': 'No current state available'}
        
        return self.impact_analyzer.analyze_impact(self.current_state, strategy_name)
    
    def generate_environment_report(self) -> Dict[str, Any]:
        """生成环境状态报告"""
        if not self.current_state:
            return {'error': 'No current state available'}
        
        state = self.current_state
        history = self.get_state_history(7)  # 最近7天
        
        # 计算统计信息
        if history:
            scores = [s.score for s in history]
            avg_score = np.mean(scores)
            score_volatility = np.std(scores)
            
            environment_distribution = {}
            for s in history:
                env_type = s.environment_type.value
                environment_distribution[env_type] = environment_distribution.get(env_type, 0) + 1
        else:
            avg_score = state.score
            score_volatility = 0.0
            environment_distribution = {state.environment_type.value: 1}
        
        # 生成报告
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'current_environment': {
                'type': state.environment_type.value,
                'trend': state.trend_direction.value,
                'score': state.score,
                'confidence': state.confidence,
                'timestamp': state.timestamp.isoformat()
            },
            'metrics_summary': {
                'volatility': state.metrics.volatility,
                'momentum': state.metrics.momentum,
                'trend_strength': state.metrics.trend_strength,
                'liquidity': state.metrics.liquidity,
                'sentiment': state.metrics.sentiment,
                'risk_level': state.metrics.risk_level
            },
            'trend_analysis': state.factors,
            'prediction': state.prediction,
            'historical_stats': {
                'average_score': avg_score,
                'score_volatility': score_volatility,
                'environment_distribution': environment_distribution,
                'data_points': len(history)
            },
            'strategy_recommendations': self._generate_strategy_recommendations(state)
        }
        
        return report
    
    def _generate_strategy_recommendations(self, state: EnvironmentState) -> Dict[str, Any]:
        """生成策略建议"""
        recommendations = {}
        
        for strategy_name in self.impact_analyzer.strategy_profiles.keys():
            impact = self.impact_analyzer.analyze_impact(state, strategy_name)
            recommendations[strategy_name] = impact
        
        return recommendations
    
    def export_state_data(self, format: str = 'json') -> str:
        """导出状态数据"""
        if not self.current_state:
            return "{}"
        
        data = {
            'current_state': {
                'timestamp': self.current_state.timestamp.isoformat(),
                'environment_type': self.current_state.environment_type.value,
                'trend_direction': self.current_state.trend_direction.value,
                'score': self.current_state.score,
                'confidence': self.current_state.confidence,
                'metrics': {
                    'volatility': self.current_state.metrics.volatility,
                    'momentum': self.current_state.metrics.momentum,
                    'trend_strength': self.current_state.metrics.trend_strength,
                    'liquidity': self.current_state.metrics.liquidity,
                    'sentiment': self.current_state.metrics.sentiment,
                    'risk_level': self.current_state.metrics.risk_level
                }
            },
            'history_count': len(self.state_history),
            'export_time': datetime.now().isoformat()
        }
        
        if format.lower() == 'json':
            return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            return str(data)
    
    def reset(self):
        """重置聚合器状态"""
        self.current_state = None
        self.state_history.clear()
        self.metrics_history.clear()
        logger.info("环境状态聚合器已重置")


# 使用示例
if __name__ == "__main__":
    # 创建聚合器
    aggregator = EnvironmentStateAggregator()
    
    # 模拟数据更新
    sample_data = {
        'price_data': {'volatility': 0.3, 'momentum': 0.2, 'trend_strength': 0.25},
        'volume_data': {'liquidity': 0.7, 'volume_ratio': 1.2},
        'volatility_data': {'volatility': 0.35},
        'sentiment_data': {'sentiment': 0.1},
        'macro_data': {'risk_level': 0.4}
    }
    
    # 更新状态
    state = aggregator.update_data_sources(sample_data)
    
    if state:
        print(f"当前环境: {state.environment_type.value}")
        print(f"趋势方向: {state.trend_direction.value}")
        print(f"综合评分: {state.score:.3f}")
        print(f"置信度: {state.confidence:.3f}")
        
        # 生成报告
        report = aggregator.generate_environment_report()
        print("\n环境状态报告:")
        print(json.dumps(report, ensure_ascii=False, indent=2))