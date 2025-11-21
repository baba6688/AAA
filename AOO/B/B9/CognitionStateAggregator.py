"""
B9认知状态聚合器
实现多模块感知结果融合、认知状态评估、置信度计算等功能
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np
from collections import deque, defaultdict
import math

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CognitiveStateLevel(Enum):
    """认知状态等级枚举"""
    CRITICAL = 1      # 危险状态
    HIGH_RISK = 2     # 高风险
    MODERATE = 3      # 中等风险
    LOW_RISK = 4      # 低风险
    STABLE = 5        # 稳定状态
    OPTIMAL = 6       # 最优状态


class ConsistencyStatus(Enum):
    """一致性状态枚举"""
    HIGH_CONSISTENCY = "high_consistency"
    MODERATE_CONSISTENCY = "moderate_consistency"
    LOW_CONSISTENCY = "low_consistency"
    INCONSISTENT = "inconsistent"


@dataclass
class PerceptionResult:
    """感知结果数据结构"""
    module_id: str
    timestamp: float
    data: Dict[str, Any]
    confidence: float
    weight: float
    category: str
    priority: int = 1
    source_reliability: float = 1.0
    data_quality: float = 1.0


@dataclass
class CognitiveState:
    """认知状态数据结构"""
    state_id: str
    timestamp: float
    level: CognitiveStateLevel
    score: float
    confidence: float
    consistency: ConsistencyStatus
    contributing_modules: List[str]
    key_indicators: Dict[str, float]
    risk_factors: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class CognitiveReport:
    """认知状态报告"""
    report_id: str
    generated_at: float
    cognitive_state: CognitiveState
    historical_trends: List[CognitiveState]
    module_performance: Dict[str, Dict[str, float]]
    consistency_analysis: Dict[str, Any]
    priority_actions: List[str]
    summary: str


class CognitionStateAggregator:
    """认知状态聚合器主类"""
    
    def __init__(self, 
                 history_size: int = 1000,
                 update_interval: float = 1.0,
                 consistency_threshold: float = 0.7,
                 confidence_threshold: float = 0.6):
        """
        初始化认知状态聚合器
        
        Args:
            history_size: 历史记录大小
            update_interval: 更新间隔（秒）
            consistency_threshold: 一致性阈值
            confidence_threshold: 置信度阈值
        """
        self.history_size = history_size
        self.update_interval = update_interval
        self.consistency_threshold = consistency_threshold
        self.confidence_threshold = confidence_threshold
        
        # 数据存储
        self.perception_results: deque = deque(maxlen=history_size)
        self.cognitive_states: deque = deque(maxlen=history_size)
        self.module_weights: Dict[str, float] = {}
        self.module_reliability: Dict[str, float] = {}
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 运行状态
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        
        # 统计信息
        self.stats = {
            'total_updates': 0,
            'successful_aggregations': 0,
            'consistency_checks': 0,
            'last_update': None
        }
        
        logger.info("认知状态聚合器初始化完成")

    def register_module(self, module_id: str, weight: float = 1.0, reliability: float = 1.0):
        """注册感知模块"""
        with self._lock:
            self.module_weights[module_id] = weight
            self.module_reliability[module_id] = reliability
            logger.info(f"注册模块: {module_id}, 权重: {weight}, 可靠性: {reliability}")

    def add_perception_result(self, result: PerceptionResult):
        """添加感知结果"""
        with self._lock:
            # 验证输入数据
            if not self._validate_perception_result(result):
                logger.warning(f"无效的感知结果: {result.module_id}")
                return False
            
            # 应用模块权重和可靠性
            result.confidence *= self.module_reliability.get(result.module_id, 1.0)
            result.weight *= self.module_weights.get(result.module_id, 1.0)
            
            self.perception_results.append(result)
            logger.debug(f"添加感知结果: {result.module_id}, 置信度: {result.confidence:.3f}")
            return True

    def _validate_perception_result(self, result: PerceptionResult) -> bool:
        """验证感知结果数据"""
        required_fields = ['module_id', 'timestamp', 'data', 'confidence', 'weight']
        
        # 检查必需字段
        for field in required_fields:
            if not hasattr(result, field) or getattr(result, field) is None:
                return False
        
        # 检查数值范围
        if not (0.0 <= result.confidence <= 1.0):
            return False
        if not (0.0 <= result.weight <= 1.0):
            return False
        if not (0.0 <= result.source_reliability <= 1.0):
            return False
        if not (0.0 <= result.data_quality <= 1.0):
            return False
            
        return True

    def fuse_perception_data(self) -> Dict[str, Any]:
        """融合感知数据"""
        with self._lock:
            if not self.perception_results:
                return {}
            
            # 按时间戳排序
            sorted_results = sorted(self.perception_results, 
                                  key=lambda x: x.timestamp, reverse=True)
            
            # 按类别分组
            categorized_data = defaultdict(list)
            for result in sorted_results:
                categorized_data[result.category].append(result)
            
            # 融合策略
            fused_data = {}
            
            for category, results in categorized_data.items():
                if not results:
                    continue
                
                # 加权平均融合
                total_weight = sum(r.weight * r.confidence * r.data_quality 
                                 for r in results)
                
                if total_weight > 0:
                    # 数据融合
                    category_data = {}
                    for result in results:
                        weight_factor = (result.weight * result.confidence * 
                                       result.data_quality) / total_weight
                        
                        for key, value in result.data.items():
                            if key not in category_data:
                                category_data[key] = 0
                            category_data[key] += value * weight_factor
                    
                    fused_data[category] = {
                        'data': category_data,
                        'confidence': min(1.0, total_weight),
                        'source_count': len(results),
                        'latest_timestamp': max(r.timestamp for r in results)
                    }
            
            logger.debug(f"数据融合完成，处理了 {len(sorted_results)} 个感知结果")
            return fused_data

    def calculate_cognitive_state(self, fused_data: Dict[str, Any]) -> CognitiveState:
        """计算认知状态"""
        if not fused_data:
            # 无数据时的默认状态
            return CognitiveState(
                state_id=f"default_{int(time.time())}",
                timestamp=time.time(),
                level=CognitiveStateLevel.MODERATE,
                score=0.5,
                confidence=0.0,
                consistency=ConsistencyStatus.LOW_CONSISTENCY,
                contributing_modules=[],
                key_indicators={},
                risk_factors=[],
                recommendations=["等待更多感知数据"],
                metadata={}
            )
        
        # 计算综合评分
        total_score = 0.0
        total_confidence = 0.0
        key_indicators = {}
        contributing_modules = set()
        
        for category, category_data in fused_data.items():
            category_score = self._calculate_category_score(category, category_data)
            category_confidence = category_data['confidence']
            
            total_score += category_score * category_confidence
            total_confidence += category_confidence
            
            key_indicators[f"{category}_score"] = category_score
            key_indicators[f"{category}_confidence"] = category_confidence
            
            # 收集贡献模块
            for result in self._get_recent_results_by_category(category):
                contributing_modules.add(result.module_id)
        
        # 归一化评分
        if total_confidence > 0:
            normalized_score = total_score / len(fused_data)
        else:
            normalized_score = 0.5
        
        # 确定认知状态等级
        level = self._determine_cognitive_level(normalized_score)
        
        # 风险因子识别
        risk_factors = self._identify_risk_factors(fused_data, normalized_score)
        
        # 生成建议
        recommendations = self._generate_recommendations(normalized_score, risk_factors)
        
        cognitive_state = CognitiveState(
            state_id=f"cog_{int(time.time())}_{hash(str(fused_data)) % 10000}",
            timestamp=time.time(),
            level=level,
            score=normalized_score,
            confidence=min(1.0, total_confidence / len(fused_data)),
            consistency=ConsistencyStatus.HIGH_CONSISTENCY,  # 将在一致性检验中更新
            contributing_modules=list(contributing_modules),
            key_indicators=key_indicators,
            risk_factors=risk_factors,
            recommendations=recommendations,
            metadata={
                'categories_processed': len(fused_data),
                'fusion_method': 'weighted_average',
                'calculation_time': time.time()
            }
        )
        
        return cognitive_state

    def _calculate_category_score(self, category: str, category_data: Dict[str, Any]) -> float:
        """计算类别评分"""
        data = category_data['data']
        confidence = category_data['confidence']
        
        # 根据不同类别使用不同的评分策略
        if category == 'market_data':
            return self._score_market_data(data)
        elif category == 'economic_indicators':
            return self._score_economic_indicators(data)
        elif category == 'sentiment_analysis':
            return self._score_sentiment_data(data)
        elif category == 'technical_analysis':
            return self._score_technical_data(data)
        else:
            # 通用评分方法
            return self._generic_score_calculation(data) * confidence

    def _score_market_data(self, data: Dict[str, Any]) -> float:
        """市场数据评分"""
        # 基于价格波动、成交量等指标评分
        volatility_score = 1.0 - min(1.0, data.get('volatility', 0.5))
        volume_score = min(1.0, data.get('volume_ratio', 1.0))
        trend_score = 0.5 + (data.get('trend_strength', 0) * 0.5)
        
        return (volatility_score + volume_score + trend_score) / 3

    def _score_economic_indicators(self, data: Dict[str, Any]) -> float:
        """经济指标评分"""
        # 基于GDP增长、通胀率、就业率等指标
        gdp_score = min(1.0, max(0.0, (data.get('gdp_growth', 0) + 5) / 10))
        inflation_score = 1.0 - min(1.0, abs(data.get('inflation_rate', 2) - 2) / 10)
        employment_score = min(1.0, data.get('employment_rate', 95) / 100)
        
        return (gdp_score + inflation_score + employment_score) / 3

    def _score_sentiment_data(self, data: Dict[str, Any]) -> float:
        """情感数据评分"""
        sentiment_score = (data.get('overall_sentiment', 0) + 1) / 2  # 归一化到[0,1]
        confidence_score = data.get('confidence', 0.5)
        volume_score = min(1.0, data.get('sentiment_volume', 1000) / 10000)
        
        return (sentiment_score + confidence_score + volume_score) / 3

    def _score_technical_data(self, data: Dict[str, Any]) -> float:
        """技术分析评分"""
        rsi_score = 1.0 - abs(data.get('rsi', 50) - 50) / 50
        macd_score = 0.5 + (data.get('macd_signal', 0) * 0.5)
        moving_avg_score = min(1.0, data.get('ma_alignment', 0.5) * 2)
        
        return (rsi_score + macd_score + moving_avg_score) / 3

    def _generic_score_calculation(self, data: Dict[str, Any]) -> float:
        """通用评分计算"""
        if not data:
            return 0.5
        
        numeric_values = [v for v in data.values() if isinstance(v, (int, float))]
        if not numeric_values:
            return 0.5
        
        # 标准化数值到[0,1]范围
        normalized_values = []
        for value in numeric_values:
            if value >= 0:
                normalized_values.append(min(1.0, value / (max(numeric_values) or 1)))
            else:
                normalized_values.append(max(0.0, 1 + value / abs(min(numeric_values) or -1)))
        
        return sum(normalized_values) / len(normalized_values)

    def _determine_cognitive_level(self, score: float) -> CognitiveStateLevel:
        """确定认知状态等级"""
        if score >= 0.9:
            return CognitiveStateLevel.OPTIMAL
        elif score >= 0.75:
            return CognitiveStateLevel.STABLE
        elif score >= 0.6:
            return CognitiveStateLevel.LOW_RISK
        elif score >= 0.4:
            return CognitiveStateLevel.MODERATE
        elif score >= 0.2:
            return CognitiveStateLevel.HIGH_RISK
        else:
            return CognitiveStateLevel.CRITICAL

    def _identify_risk_factors(self, fused_data: Dict[str, Any], score: float) -> List[str]:
        """识别风险因子"""
        risk_factors = []
        
        # 基于评分识别风险
        if score < 0.3:
            risk_factors.append("综合评分过低")
        elif score < 0.5:
            risk_factors.append("综合评分偏低")
        
        # 基于数据一致性识别风险
        consistency = self.check_cognitive_consistency()
        if consistency == ConsistencyStatus.INCONSISTENT:
            risk_factors.append("感知数据不一致")
        elif consistency == ConsistencyStatus.LOW_CONSISTENCY:
            risk_factors.append("感知数据一致性较低")
        
        # 基于置信度识别风险
        low_confidence_categories = []
        for category, category_data in fused_data.items():
            if category_data['confidence'] < self.confidence_threshold:
                low_confidence_categories.append(category)
        
        if low_confidence_categories:
            risk_factors.append(f"低置信度类别: {', '.join(low_confidence_categories)}")
        
        # 基于数据新鲜度识别风险
        stale_data_modules = self._identify_stale_data()
        if stale_data_modules:
            risk_factors.append(f"数据过时模块: {', '.join(stale_data_modules)}")
        
        return risk_factors

    def _generate_recommendations(self, score: float, risk_factors: List[str]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于评分生成建议
        if score >= 0.8:
            recommendations.append("当前状态良好，维持现有策略")
            recommendations.append("可考虑适度增加投资")
        elif score >= 0.6:
            recommendations.append("状态稳定，谨慎操作")
            recommendations.append("加强监控关键指标")
        elif score >= 0.4:
            recommendations.append("状态一般，建议减仓观望")
            recommendations.append("等待更明确的市场信号")
        else:
            recommendations.append("状态不佳，建议清仓避险")
            recommendations.append("重新评估投资策略")
        
        # 基于风险因子生成建议
        for risk in risk_factors:
            if "不一致" in risk:
                recommendations.append("检查数据源，确保信息一致性")
            if "低置信度" in risk:
                recommendations.append("提升数据收集质量和频率")
            if "过时" in risk:
                recommendations.append("更新数据源，确保数据时效性")
        
        return recommendations

    def _get_recent_results_by_category(self, category: str) -> List[PerceptionResult]:
        """获取指定类别的最近感知结果"""
        cutoff_time = time.time() - 300  # 5分钟内的数据
        return [r for r in self.perception_results 
                if r.category == category and r.timestamp > cutoff_time]

    def _identify_stale_data(self) -> List[str]:
        """识别数据过时的模块"""
        stale_modules = []
        cutoff_time = time.time() - 600  # 10分钟
        
        recent_modules = set()
        for result in self.perception_results:
            if result.timestamp > cutoff_time:
                recent_modules.add(result.module_id)
        
        for module_id in self.module_weights:
            if module_id not in recent_modules:
                stale_modules.append(module_id)
        
        return stale_modules

    def check_cognitive_consistency(self) -> ConsistencyStatus:
        """检查认知一致性"""
        if len(self.perception_results) < 2:
            return ConsistencyStatus.MODERATE_CONSISTENCY
        
        # 按模块分组最近的感知结果
        module_results = defaultdict(list)
        cutoff_time = time.time() - 300  # 5分钟内
        
        for result in self.perception_results:
            if result.timestamp > cutoff_time:
                module_results[result.module_id].append(result)
        
        # 计算模块间一致性
        consistency_scores = []
        
        module_list = list(module_results.keys())
        for i in range(len(module_list)):
            for j in range(i + 1, len(module_list)):
                module1, module2 = module_list[i], module_list[j]
                
                if module_results[module1] and module_results[module2]:
                    # 获取最新结果
                    latest1 = max(module_results[module1], key=lambda x: x.timestamp)
                    latest2 = max(module_results[module2], key=lambda x: x.timestamp)
                    
                    # 计算一致性
                    consistency = self._calculate_module_consistency(latest1, latest2)
                    consistency_scores.append(consistency)
        
        if not consistency_scores:
            return ConsistencyStatus.MODERATE_CONSISTENCY
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        
        if avg_consistency >= 0.8:
            return ConsistencyStatus.HIGH_CONSISTENCY
        elif avg_consistency >= 0.6:
            return ConsistencyStatus.MODERATE_CONSISTENCY
        elif avg_consistency >= 0.4:
            return ConsistencyStatus.LOW_CONSISTENCY
        else:
            return ConsistencyStatus.INCONSISTENT

    def _calculate_module_consistency(self, result1: PerceptionResult, 
                                    result2: PerceptionResult) -> float:
        """计算两个模块结果的一致性"""
        # 时间一致性
        time_diff = abs(result1.timestamp - result2.timestamp)
        time_consistency = max(0, 1 - time_diff / 300)  # 5分钟窗口
        
        # 数据一致性（简化版本）
        data_consistency = 1.0  # 实际实现中需要更复杂的相似度计算
        
        # 置信度一致性
        conf_diff = abs(result1.confidence - result2.confidence)
        confidence_consistency = max(0, 1 - conf_diff)
        
        # 综合一致性
        return (time_consistency + data_consistency + confidence_consistency) / 3

    def calculate_perception_confidence(self, fused_data: Dict[str, Any]) -> float:
        """计算感知置信度"""
        if not fused_data:
            return 0.0
        
        confidence_scores = []
        
        for category, category_data in fused_data.items():
            base_confidence = category_data['confidence']
            source_count = category_data['source_count']
            
            # 基于数据源数量调整置信度
            source_bonus = min(0.2, (source_count - 1) * 0.05)
            
            # 基于数据新鲜度调整置信度
            freshness = self._calculate_data_freshness(category_data['latest_timestamp'])
            
            adjusted_confidence = (base_confidence + source_bonus) * freshness
            confidence_scores.append(min(1.0, adjusted_confidence))
        
        return sum(confidence_scores) / len(confidence_scores)

    def _calculate_data_freshness(self, timestamp: float) -> float:
        """计算数据新鲜度"""
        age = time.time() - timestamp
        if age <= 60:  # 1分钟内
            return 1.0
        elif age <= 300:  # 5分钟内
            return 0.8
        elif age <= 900:  # 15分钟内
            return 0.6
        else:
            return 0.3

    def prioritize_perception_results(self, limit: int = 10) -> List[PerceptionResult]:
        """感知结果优先级排序"""
        with self._lock:
            if not self.perception_results:
                return []
            
            # 计算优先级分数
            scored_results = []
            for result in self.perception_results:
                priority_score = self._calculate_priority_score(result)
                scored_results.append((priority_score, result))
            
            # 按优先级排序
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # 返回前N个结果
            return [result for _, result in scored_results[:limit]]

    def _calculate_priority_score(self, result: PerceptionResult) -> float:
        """计算感知结果优先级分数"""
        # 基础分数
        base_score = result.confidence * result.weight
        
        # 时间衰减因子
        age = time.time() - result.timestamp
        time_factor = max(0.1, 1.0 - age / 3600)  # 1小时内线性衰减
        
        # 优先级权重
        priority_factor = result.priority / 5.0  # 假设最高优先级为5
        
        # 数据质量因子
        quality_factor = result.data_quality
        
        # 模块可靠性因子
        reliability_factor = self.module_reliability.get(result.module_id, 1.0)
        
        return base_score * time_factor * priority_factor * quality_factor * reliability_factor

    def update_cognitive_state(self) -> CognitiveState:
        """更新认知状态"""
        self.stats['total_updates'] += 1
        
        try:
            # 融合感知数据
            fused_data = self.fuse_perception_data()
            
            # 计算认知状态
            cognitive_state = self.calculate_cognitive_state(fused_data)
            
            # 检查一致性并更新状态
            consistency = self.check_cognitive_consistency()
            cognitive_state.consistency = consistency
            
            # 计算感知置信度
            perception_confidence = self.calculate_perception_confidence(fused_data)
            cognitive_state.confidence = min(cognitive_state.confidence, perception_confidence)
            
            # 添加到历史记录
            self.cognitive_states.append(cognitive_state)
            
            self.stats['successful_aggregations'] += 1
            self.stats['last_update'] = time.time()
            
            logger.info(f"认知状态更新完成: {cognitive_state.level.name}, "
                       f"评分: {cognitive_state.score:.3f}")
            
            return cognitive_state
            
        except Exception as e:
            logger.error(f"认知状态更新失败: {e}")
            raise

    def get_historical_states(self, hours: int = 24) -> List[CognitiveState]:
        """获取历史认知状态"""
        cutoff_time = time.time() - (hours * 3600)
        return [state for state in self.cognitive_states 
                if state.timestamp > cutoff_time]

    def generate_cognitive_report(self, hours: int = 24) -> CognitiveReport:
        """生成认知状态报告"""
        current_state = self.update_cognitive_state()
        historical_states = self.get_historical_states(hours)
        
        # 模块性能分析
        module_performance = self._analyze_module_performance()
        
        # 一致性分析
        consistency_analysis = self._analyze_consistency_trends(historical_states)
        
        # 优先级行动
        priority_actions = self._generate_priority_actions(current_state)
        
        # 生成摘要
        summary = self._generate_report_summary(current_state, historical_states)
        
        report = CognitiveReport(
            report_id=f"cog_report_{int(time.time())}",
            generated_at=time.time(),
            cognitive_state=current_state,
            historical_trends=historical_states,
            module_performance=module_performance,
            consistency_analysis=consistency_analysis,
            priority_actions=priority_actions,
            summary=summary
        )
        
        logger.info(f"认知状态报告生成完成: {report.report_id}")
        return report

    def _analyze_module_performance(self) -> Dict[str, Dict[str, float]]:
        """分析模块性能"""
        performance = {}
        
        for module_id in self.module_weights:
            module_results = [r for r in self.perception_results 
                            if r.module_id == module_id]
            
            if module_results:
                avg_confidence = sum(r.confidence for r in module_results) / len(module_results)
                avg_weight = sum(r.weight for r in module_results) / len(module_results)
                data_quality = sum(r.data_quality for r in module_results) / len(module_results)
                reliability = self.module_reliability.get(module_id, 1.0)
                
                performance[module_id] = {
                    'average_confidence': avg_confidence,
                    'average_weight': avg_weight,
                    'data_quality': data_quality,
                    'reliability': reliability,
                    'total_results': len(module_results),
                    'latest_timestamp': max(r.timestamp for r in module_results)
                }
            else:
                performance[module_id] = {
                    'average_confidence': 0.0,
                    'average_weight': 0.0,
                    'data_quality': 0.0,
                    'reliability': self.module_reliability.get(module_id, 1.0),
                    'total_results': 0,
                    'latest_timestamp': 0.0
                }
        
        return performance

    def _analyze_consistency_trends(self, historical_states: List[CognitiveState]) -> Dict[str, Any]:
        """分析一致性趋势"""
        if not historical_states:
            return {}
        
        consistency_counts = defaultdict(int)
        consistency_scores = []
        
        for state in historical_states:
            consistency_counts[state.consistency.value] += 1
            # 将一致性状态转换为数值分数
            score_map = {
                ConsistencyStatus.HIGH_CONSISTENCY: 1.0,
                ConsistencyStatus.MODERATE_CONSISTENCY: 0.7,
                ConsistencyStatus.LOW_CONSISTENCY: 0.4,
                ConsistencyStatus.INCONSISTENT: 0.1
            }
            consistency_scores.append(score_map[state.consistency])
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        
        return {
            'consistency_distribution': dict(consistency_counts),
            'average_consistency_score': avg_consistency,
            'consistency_trend': 'improving' if consistency_scores[-1] > consistency_scores[0] else 'declining',
            'total_states_analyzed': len(historical_states)
        }

    def _generate_priority_actions(self, current_state: CognitiveState) -> List[str]:
        """生成优先级行动"""
        actions = []
        
        # 基于当前状态等级的行动
        if current_state.level in [CognitiveStateLevel.CRITICAL, CognitiveStateLevel.HIGH_RISK]:
            actions.append("立即执行风险控制措施")
            actions.append("暂停所有高风险操作")
            actions.append("启动应急响应程序")
        elif current_state.level == CognitiveStateLevel.MODERATE:
            actions.append("加强监控频率")
            actions.append("准备风险对冲策略")
        elif current_state.level == CognitiveStateLevel.LOW_RISK:
            actions.append("维持现有策略")
            actions.append("关注市场变化")
        else:  # STABLE, OPTIMAL
            actions.append("优化资源配置")
            actions.append("考虑扩张机会")
        
        # 基于一致性的行动
        if current_state.consistency == ConsistencyStatus.INCONSISTENT:
            actions.append("立即解决数据不一致问题")
        elif current_state.consistency == ConsistencyStatus.LOW_CONSISTENCY:
            actions.append("检查数据源质量")
        
        # 基于置信度的行动
        if current_state.confidence < self.confidence_threshold:
            actions.append("提高数据收集质量")
            actions.append("增加验证机制")
        
        return actions

    def _generate_report_summary(self, current_state: CognitiveState, 
                               historical_states: List[CognitiveState]) -> str:
        """生成报告摘要"""
        summary_parts = []
        
        # 当前状态摘要
        summary_parts.append(f"当前认知状态: {current_state.level.name}")
        summary_parts.append(f"综合评分: {current_state.score:.3f}")
        summary_parts.append(f"置信度: {current_state.confidence:.3f}")
        summary_parts.append(f"一致性: {current_state.consistency.value}")
        
        # 趋势分析
        if len(historical_states) >= 2:
            recent_score = historical_states[-1].score
            previous_score = historical_states[-2].score
            trend = "上升" if recent_score > previous_score else "下降"
            summary_parts.append(f"评分趋势: {trend}")
        
        # 风险提示
        if current_state.risk_factors:
            summary_parts.append(f"主要风险: {', '.join(current_state.risk_factors[:3])}")
        
        # 建议摘要
        if current_state.recommendations:
            summary_parts.append(f"主要建议: {current_state.recommendations[0]}")
        
        return "; ".join(summary_parts)

    def start_real_time_updates(self):
        """启动实时更新"""
        if self._running:
            logger.warning("实时更新已在运行中")
            return
        
        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        logger.info("实时更新已启动")

    def stop_real_time_updates(self):
        """停止实时更新"""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5)
        logger.info("实时更新已停止")

    def _update_loop(self):
        """更新循环"""
        while self._running:
            try:
                self.update_cognitive_state()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"更新循环错误: {e}")
                time.sleep(self.update_interval)

    def export_state_data(self, filepath: str):
        """导出状态数据"""
        data = {
            'cognitive_states': [asdict(state) for state in self.cognitive_states],
            'perception_results': [asdict(result) for result in self.perception_results],
            'module_weights': self.module_weights,
            'module_reliability': self.module_reliability,
            'statistics': self.stats,
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"状态数据已导出到: {filepath}")

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'running': self._running,
            'total_perception_results': len(self.perception_results),
            'total_cognitive_states': len(self.cognitive_states),
            'registered_modules': list(self.module_weights.keys()),
            'statistics': self.stats.copy(),
            'configuration': {
                'history_size': self.history_size,
                'update_interval': self.update_interval,
                'consistency_threshold': self.consistency_threshold,
                'confidence_threshold': self.confidence_threshold
            }
        }


# 便利函数
def create_sample_perception_result(module_id: str, category: str, 
                                  data: Dict[str, Any]) -> PerceptionResult:
    """创建示例感知结果"""
    return PerceptionResult(
        module_id=module_id,
        timestamp=time.time(),
        data=data,
        confidence=np.random.uniform(0.6, 1.0),
        weight=np.random.uniform(0.7, 1.0),
        category=category,
        priority=np.random.randint(1, 5),
        source_reliability=np.random.uniform(0.8, 1.0),
        data_quality=np.random.uniform(0.7, 1.0)
    )


if __name__ == "__main__":
    # 示例用法
    aggregator = CognitionStateAggregator()
    
    # 注册模块
    aggregator.register_module("market_data", weight=1.0, reliability=0.9)
    aggregator.register_module("sentiment_analysis", weight=0.8, reliability=0.8)
    aggregator.register_module("technical_analysis", weight=0.9, reliability=0.85)
    
    # 添加示例感知结果
    sample_results = [
        create_sample_perception_result("market_data", "market_data", 
                                      {"volatility": 0.3, "volume_ratio": 1.2, "trend_strength": 0.6}),
        create_sample_perception_result("sentiment_analysis", "sentiment_analysis",
                                      {"overall_sentiment": 0.2, "confidence": 0.8, "sentiment_volume": 5000}),
        create_sample_perception_result("technical_analysis", "technical_analysis",
                                      {"rsi": 45, "macd_signal": 0.1, "ma_alignment": 0.7})
    ]
    
    for result in sample_results:
        aggregator.add_perception_result(result)
    
    # 更新认知状态
    cognitive_state = aggregator.update_cognitive_state()
    print(f"当前认知状态: {cognitive_state.level.name}")
    print(f"综合评分: {cognitive_state.score:.3f}")
    print(f"置信度: {cognitive_state.confidence:.3f}")
    print(f"一致性: {cognitive_state.consistency.value}")
    
    # 生成报告
    report = aggregator.generate_cognitive_report()
    print(f"\n报告摘要: {report.summary}")
    print(f"优先级行动: {', '.join(report.priority_actions)}")
    
    # 导出数据
    aggregator.export_state_data("cognitive_state_data.json")
    print("\n数据已导出到 cognitive_state_data.json")