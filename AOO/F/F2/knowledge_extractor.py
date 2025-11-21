"""
策略知识提取器模块

实现策略知识的提取、管理和应用：
- 策略模式识别
- 知识图谱构建
- 规则提取
- 洞察生成
- 知识验证
- 知识库管理
- 知识推理
- 经验总结
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging
from copy import deepcopy
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

from .StrategyLearner import BaseStrategy, StrategyType, StrategyPerformance, LearningContext

logger = logging.getLogger(__name__)

@dataclass
class KnowledgePattern:
    """知识模式数据类"""
    pattern_id: str
    pattern_type: str
    description: str
    confidence: float
    support_count: int
    examples: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None

@dataclass
class KnowledgeRule:
    """知识规则数据类"""
    rule_id: str
    rule_type: str
    condition: str
    conclusion: str
    confidence: float
    strength: float
    source_patterns: List[str]
    validation_results: Dict[str, Any] = None

@dataclass
class KnowledgeInsight:
    """知识洞察数据类"""
    insight_id: str
    category: str
    title: str
    description: str
    importance: float
    actionable: bool
    supporting_evidence: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = None

@dataclass
class KnowledgeNode:
    """知识图谱节点"""
    node_id: str
    node_type: str
    label: str
    properties: Dict[str, Any]
    connections: List[str] = None

@dataclass
class KnowledgeEdge:
    """知识图谱边"""
    edge_id: str
    source_node: str
    target_node: str
    relationship_type: str
    strength: float
    metadata: Dict[str, Any] = None

class PatternRecognizer:
    """模式识别器"""
    
    def __init__(self):
        self.patterns = []
        self.min_support = 3
        self.min_confidence = 0.6
        
    def recognize_patterns(self, strategies: List[BaseStrategy], 
                          performance_data: Dict[str, List[StrategyPerformance]]) -> List[KnowledgePattern]:
        """识别策略模式"""
        try:
            patterns = []
            
            # 策略类型模式
            type_patterns = self._recognize_type_patterns(strategies)
            patterns.extend(type_patterns)
            
            # 性能模式
            performance_patterns = self._recognize_performance_patterns(performance_data)
            patterns.extend(performance_patterns)
            
            # 行为模式
            behavior_patterns = self._recognize_behavior_patterns(strategies, performance_data)
            patterns.extend(behavior_patterns)
            
            # 时序模式
            temporal_patterns = self._recognize_temporal_patterns(performance_data)
            patterns.extend(temporal_patterns)
            
            # 关联模式
            correlation_patterns = self._recognize_correlation_patterns(performance_data)
            patterns.extend(correlation_patterns)
            
            self.patterns = patterns
            logger.info(f"识别了 {len(patterns)} 个策略模式")
            
            return patterns
            
        except Exception as e:
            logger.error(f"模式识别出错: {e}")
            return []
    
    def _recognize_type_patterns(self, strategies: List[BaseStrategy]) -> List[KnowledgePattern]:
        """识别策略类型模式"""
        patterns = []
        
        # 统计策略类型分布
        type_counts = Counter(strategy.strategy_type.value for strategy in strategies)
        total_strategies = len(strategies)
        
        for strategy_type, count in type_counts.items():
            confidence = count / total_strategies
            
            if confidence >= self.min_confidence and count >= self.min_support:
                pattern = KnowledgePattern(
                    pattern_id=f"type_pattern_{strategy_type}",
                    pattern_type="strategy_type_distribution",
                    description=f"策略组合中 {strategy_type} 类型策略占 {confidence:.2%}",
                    confidence=confidence,
                    support_count=count,
                    examples=[{"strategy_type": strategy_type, "count": count}],
                    metadata={
                        "total_strategies": total_strategies,
                        "type_distribution": dict(type_counts)
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _recognize_performance_patterns(self, performance_data: Dict[str, List[StrategyPerformance]]) -> List[KnowledgePattern]:
        """识别性能模式"""
        patterns = []
        
        for strategy_id, performances in performance_data.items():
            if len(performances) < 5:
                continue
            
            returns = [p.return_rate for p in performances]
            win_rates = [p.win_rate for p in performances]
            
            # 高胜率模式
            avg_win_rate = np.mean(win_rates)
            if avg_win_rate >= 0.7:
                pattern = KnowledgePattern(
                    pattern_id=f"high_winrate_{strategy_id}",
                    pattern_type="performance_excellence",
                    description=f"策略 {strategy_id} 具有高胜率 ({avg_win_rate:.2%})",
                    confidence=avg_win_rate,
                    support_count=len(performances),
                    examples=[{"strategy_id": strategy_id, "avg_win_rate": avg_win_rate}],
                    metadata={"performance_metric": "win_rate"}
                )
                patterns.append(pattern)
            
            # 高收益模式
            avg_return = np.mean(returns)
            if avg_return >= 0.02:  # 2%平均收益
                pattern = KnowledgePattern(
                    pattern_id=f"high_return_{strategy_id}",
                    pattern_type="performance_excellence",
                    description=f"策略 {strategy_id} 具有高收益 ({avg_return:.2%})",
                    confidence=min(1.0, avg_return / 0.1),  # 归一化到[0,1]
                    support_count=len(performances),
                    examples=[{"strategy_id": strategy_id, "avg_return": avg_return}],
                    metadata={"performance_metric": "return_rate"}
                )
                patterns.append(pattern)
            
            # 稳定性模式
            return_std = np.std(returns)
            if return_std <= 0.05:  # 低波动性
                stability_score = 1.0 - (return_std / 0.1)  # 波动性越低得分越高
                pattern = KnowledgePattern(
                    pattern_id=f"stability_{strategy_id}",
                    pattern_type="performance_stability",
                    description=f"策略 {strategy_id} 具有高稳定性 (波动率: {return_std:.3f})",
                    confidence=stability_score,
                    support_count=len(performances),
                    examples=[{"strategy_id": strategy_id, "volatility": return_std}],
                    metadata={"performance_metric": "stability"}
                )
                patterns.append(pattern)
        
        return patterns
    
    def _recognize_behavior_patterns(self, strategies: List[BaseStrategy], 
                                   performance_data: Dict[str, List[StrategyPerformance]]) -> List[KnowledgePattern]:
        """识别行为模式"""
        patterns = []
        
        # 策略使用频率模式
        usage_counts = {}
        for strategy in strategies:
            usage_counts[strategy.strategy_id] = strategy.state.usage_count
        
        # 高频使用策略
        if usage_counts:
            max_usage = max(usage_counts.values())
            for strategy_id, usage in usage_counts.items():
                if usage >= max_usage * 0.8:  # 使用频率在前20%
                    confidence = usage / max_usage
                    pattern = KnowledgePattern(
                        pattern_id=f"high_usage_{strategy_id}",
                        pattern_type="behavior_frequency",
                        description=f"策略 {strategy_id} 被高频使用 (使用次数: {usage})",
                        confidence=confidence,
                        support_count=usage,
                        examples=[{"strategy_id": strategy_id, "usage_count": usage}],
                        metadata={"behavior_metric": "usage_frequency"}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _recognize_temporal_patterns(self, performance_data: Dict[str, List[StrategyPerformance]]) -> List[KnowledgePattern]:
        """识别时序模式"""
        patterns = []
        
        for strategy_id, performances in performance_data.items():
            if len(performances) < 10:
                continue
            
            # 按时间排序
            sorted_performances = sorted(performances, key=lambda x: x.timestamp)
            returns = [p.return_rate for p in sorted_performances]
            
            # 趋势模式
            if len(returns) >= 20:
                # 计算移动平均
                ma_10 = np.convolve(returns, np.ones(10)/10, mode='valid')
                ma_20 = np.convolve(returns, np.ones(20)/20, mode='valid')
                
                if len(ma_10) > 0 and len(ma_20) > 0:
                    # 金叉模式
                    golden_crosses = 0
                    for i in range(1, min(len(ma_10), len(ma_20))):
                        if ma_10[i] > ma_20[i] and ma_10[i-1] <= ma_20[i-1]:
                            golden_crosses += 1
                    
                    if golden_crosses > 0:
                        pattern = KnowledgePattern(
                            pattern_id=f"golden_cross_{strategy_id}",
                            pattern_type="temporal_trend",
                            description=f"策略 {strategy_id} 出现 {golden_crosses} 次金叉信号",
                            confidence=min(1.0, golden_crosses / 10),
                            support_count=golden_crosses,
                            examples=[{"strategy_id": strategy_id, "golden_crosses": golden_crosses}],
                            metadata={"temporal_metric": "trend_alignment"}
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _recognize_correlation_patterns(self, performance_data: Dict[str, List[StrategyPerformance]]) -> List[KnowledgePattern]:
        """识别关联模式"""
        patterns = []
        
        strategy_ids = list(performance_data.keys())
        
        for i, strategy1 in enumerate(strategy_ids):
            for j, strategy2 in enumerate(strategy_ids[i+1:], i+1):
                returns1 = [p.return_rate for p in performance_data[strategy1]]
                returns2 = [p.return_rate for p in performance_data[strategy2]]
                
                min_length = min(len(returns1), len(returns2))
                if min_length < 10:
                    continue
                
                # 计算相关性
                correlation = np.corrcoef(returns1[:min_length], returns2[:min_length])[0, 1]
                
                if not np.isnan(correlation):
                    # 高正相关模式
                    if correlation >= 0.7:
                        pattern = KnowledgePattern(
                            pattern_id=f"high_positive_corr_{strategy1}_{strategy2}",
                            pattern_type="correlation",
                            description=f"策略 {strategy1} 和 {strategy2} 高度正相关 (相关系数: {correlation:.3f})",
                            confidence=correlation,
                            support_count=min_length,
                            examples=[{"strategy1": strategy1, "strategy2": strategy2, "correlation": correlation}],
                            metadata={"correlation_type": "positive_high"}
                        )
                        patterns.append(pattern)
                    
                    # 高负相关模式
                    elif correlation <= -0.7:
                        pattern = KnowledgePattern(
                            pattern_id=f"high_negative_corr_{strategy1}_{strategy2}",
                            pattern_type="correlation",
                            description=f"策略 {strategy1} 和 {strategy2} 高度负相关 (相关系数: {correlation:.3f})",
                            confidence=abs(correlation),
                            support_count=min_length,
                            examples=[{"strategy1": strategy1, "strategy2": strategy2, "correlation": correlation}],
                            metadata={"correlation_type": "negative_high"}
                        )
                        patterns.append(pattern)
        
        return patterns

class RuleExtractor:
    """规则提取器"""
    
    def __init__(self):
        self.rules = []
        self.min_rule_support = 3
        
    def extract_rules(self, patterns: List[KnowledgePattern]) -> List[KnowledgeRule]:
        """从模式中提取规则"""
        try:
            rules = []
            
            # 基于模式的关联规则
            association_rules = self._extract_association_rules(patterns)
            rules.extend(association_rules)
            
            # 性能规则
            performance_rules = self._extract_performance_rules(patterns)
            rules.extend(performance_rules)
            
            # 行为规则
            behavior_rules = self._extract_behavior_rules(patterns)
            rules.extend(behavior_rules)
            
            # 条件规则
            conditional_rules = self._extract_conditional_rules(patterns)
            rules.extend(conditional_rules)
            
            self.rules = rules
            logger.info(f"提取了 {len(rules)} 条知识规则")
            
            return rules
            
        except Exception as e:
            logger.error(f"规则提取出错: {e}")
            return []
    
    def _extract_association_rules(self, patterns: List[KnowledgePattern]) -> List[KnowledgeRule]:
        """提取关联规则"""
        rules = []
        
        # 按模式类型分组
        patterns_by_type = defaultdict(list)
        for pattern in patterns:
            patterns_by_type[pattern.pattern_type].append(pattern)
        
        # 生成类型间关联规则
        for type1, patterns1 in patterns_by_type.items():
            for type2, patterns2 in patterns_by_type.items():
                if type1 != type2:
                    # 计算共同支持度
                    common_examples = 0
                    total_examples = 0
                    
                    for p1 in patterns1:
                        for p2 in patterns2:
                            # 检查是否有共同的支持例子
                            examples1 = {json.dumps(ex, sort_keys=True) for ex in p1.examples}
                            examples2 = {json.dumps(ex, sort_keys=True) for ex in p2.examples}
                            common_examples += len(examples1 & examples2)
                            total_examples += len(examples1 | examples2)
                    
                    if total_examples > 0:
                        support = common_examples / total_examples
                        if support >= 0.3:  # 最小支持度
                            rule = KnowledgeRule(
                                rule_id=f"assoc_{type1}_{type2}",
                                rule_type="association",
                                condition=f"存在 {type1} 模式",
                                conclusion=f"很可能也存在 {type2} 模式",
                                confidence=support,
                                strength=support,
                                source_patterns=[p.pattern_id for p in patterns1[:3]]  # 取前3个作为来源
                            )
                            rules.append(rule)
        
        return rules
    
    def _extract_performance_rules(self, patterns: List[KnowledgePattern]) -> List[KnowledgeRule]:
        """提取性能规则"""
        rules = []
        
        # 高胜率策略规则
        high_winrate_patterns = [p for p in patterns if "high_winrate" in p.pattern_id]
        if high_winrate_patterns:
            rule = KnowledgeRule(
                rule_id="high_winrate_rule",
                rule_type="performance",
                condition="策略具有高胜率 (>= 70%)",
                conclusion="该策略在当前市场环境下表现良好",
                confidence=np.mean([p.confidence for p in high_winrate_patterns]),
                strength=0.8,
                source_patterns=[p.pattern_id for p in high_winrate_patterns]
            )
            rules.append(rule)
        
        # 高收益策略规则
        high_return_patterns = [p for p in patterns if "high_return" in p.pattern_id]
        if high_return_patterns:
            rule = KnowledgeRule(
                rule_id="high_return_rule",
                rule_type="performance",
                condition="策略具有高收益 (>= 2%)",
                conclusion="该策略具有良好的盈利能力",
                confidence=np.mean([p.confidence for p in high_return_patterns]),
                strength=0.75,
                source_patterns=[p.pattern_id for p in high_return_patterns]
            )
            rules.append(rule)
        
        # 稳定性规则
        stability_patterns = [p for p in patterns if "stability" in p.pattern_id]
        if stability_patterns:
            rule = KnowledgeRule(
                rule_id="stability_rule",
                rule_type="performance",
                condition="策略具有高稳定性 (低波动率)",
                conclusion="该策略风险控制良好，适合稳健投资",
                confidence=np.mean([p.confidence for p in stability_patterns]),
                strength=0.7,
                source_patterns=[p.pattern_id for p in stability_patterns]
            )
            rules.append(rule)
        
        return rules
    
    def _extract_behavior_rules(self, patterns: List[KnowledgePattern]) -> List[KnowledgeRule]:
        """提取行为规则"""
        rules = []
        
        # 高频使用规则
        usage_patterns = [p for p in patterns if "high_usage" in p.pattern_id]
        if usage_patterns:
            rule = KnowledgeRule(
                rule_id="high_usage_rule",
                rule_type="behavior",
                condition="策略被高频使用",
                conclusion="该策略在实践中证明有效",
                confidence=np.mean([p.confidence for p in usage_patterns]),
                strength=0.6,
                source_patterns=[p.pattern_id for p in usage_patterns]
            )
            rules.append(rule)
        
        return rules
    
    def _extract_conditional_rules(self, patterns: List[KnowledgePattern]) -> List[KnowledgeRule]:
        """提取条件规则"""
        rules = []
        
        # 相关性规则
        corr_patterns = [p for p in patterns if p.pattern_type == "correlation"]
        for pattern in corr_patterns:
            if "positive" in pattern.metadata.get("correlation_type", ""):
                rule = KnowledgeRule(
                    rule_id=f"positive_corr_{pattern.pattern_id}",
                    rule_type="correlation",
                    condition=f"策略间高度正相关 (相关系数 >= 0.7)",
                    conclusion="这些策略可能面临相似的市场风险",
                    confidence=pattern.confidence,
                    strength=0.8,
                    source_patterns=[pattern.pattern_id]
                )
                rules.append(rule)
            elif "negative" in pattern.metadata.get("correlation_type", ""):
                rule = KnowledgeRule(
                    rule_id=f"negative_corr_{pattern.pattern_id}",
                    rule_type="correlation",
                    condition=f"策略间高度负相关 (相关系数 <= -0.7)",
                    conclusion="这些策略可以实现风险对冲",
                    confidence=pattern.confidence,
                    strength=0.9,
                    source_patterns=[pattern.pattern_id]
                )
                rules.append(rule)
        
        return rules

class InsightGenerator:
    """洞察生成器"""
    
    def __init__(self):
        self.insights = []
        
    def generate_insights(self, patterns: List[KnowledgePattern], 
                         rules: List[KnowledgeRule],
                         strategies: List[BaseStrategy],
                         performance_data: Dict[str, List[StrategyPerformance]]) -> List[KnowledgeInsight]:
        """生成知识洞察"""
        try:
            insights = []
            
            # 策略组合洞察
            portfolio_insights = self._generate_portfolio_insights(strategies, patterns)
            insights.extend(portfolio_insights)
            
            # 性能洞察
            performance_insights = self._generate_performance_insights(patterns, performance_data)
            insights.extend(performance_insights)
            
            # 优化洞察
            optimization_insights = self._generate_optimization_insights(patterns, rules)
            insights.extend(optimization_insights)
            
            # 风险洞察
            risk_insights = self._generate_risk_insights(patterns, performance_data)
            insights.extend(risk_insights)
            
            # 市场洞察
            market_insights = self._generate_market_insights(strategies, patterns)
            insights.extend(market_insights)
            
            self.insights = insights
            logger.info(f"生成了 {len(insights)} 个知识洞察")
            
            return insights
            
        except Exception as e:
            logger.error(f"洞察生成出错: {e}")
            return []
    
    def _generate_portfolio_insights(self, strategies: List[BaseStrategy], 
                                   patterns: List[KnowledgePattern]) -> List[KnowledgeInsight]:
        """生成组合洞察"""
        insights = []
        
        # 策略多样性洞察
        strategy_types = set(strategy.strategy_type for strategy in strategies)
        if len(strategy_types) >= 3:
            insight = KnowledgeInsight(
                insight_id="portfolio_diversity",
                category="portfolio_management",
                title="策略组合具有良好的多样性",
                description=f"当前组合包含 {len(strategy_types)} 种不同类型的策略，有助于提高整体稳健性",
                importance=0.8,
                actionable=True,
                supporting_evidence=[f"策略类型: {[t.value for t in strategy_types]}"],
                recommendations=[
                    "保持策略多样性",
                    "定期评估各策略类型的贡献度",
                    "考虑引入更多类型的策略"
                ],
                metadata={"strategy_types": [t.value for t in strategy_types]}
            )
            insights.append(insight)
        
        # 策略分布洞察
        type_distribution = Counter(strategy.strategy_type.value for strategy in strategies)
        dominant_type = type_distribution.most_common(1)[0]
        
        if dominant_type[1] / len(strategies) > 0.6:  # 单一类型超过60%
            insight = KnowledgeInsight(
                insight_id="portfolio_concentration",
                category="portfolio_management",
                title="策略组合存在类型集中风险",
                description=f"{dominant_type[0]} 类型策略占比 {dominant_type[1]/len(strategies):.1%}，可能存在集中风险",
                importance=0.7,
                actionable=True,
                supporting_evidence=[f"类型分布: {dict(type_distribution)}"],
                recommendations=[
                    "考虑增加其他类型策略的权重",
                    "分散化投资降低集中风险",
                    "定期重新平衡策略组合"
                ],
                metadata={"type_distribution": dict(type_distribution)}
            )
            insights.append(insight)
        
        return insights
    
    def _generate_performance_insights(self, patterns: List[KnowledgePattern],
                                     performance_data: Dict[str, List[StrategyPerformance]]) -> List[KnowledgeInsight]:
        """生成性能洞察"""
        insights = []
        
        # 最佳策略识别
        performance_patterns = [p for p in patterns if p.pattern_type in ["performance_excellence", "performance_stability"]]
        if performance_patterns:
            # 按置信度排序
            performance_patterns.sort(key=lambda x: x.confidence, reverse=True)
            best_pattern = performance_patterns[0]
            
            insight = KnowledgeInsight(
                insight_id="best_performing_strategy",
                category="performance",
                title="识别出最佳表现策略",
                description=f"基于分析，{best_pattern.description}，置信度为 {best_pattern.confidence:.2%}",
                importance=0.9,
                actionable=True,
                supporting_evidence=[best_pattern.pattern_id],
                recommendations=[
                    "重点关注该策略的表现",
                    "考虑增加其权重",
                    "研究其成功因素并应用到其他策略"
                ],
                metadata={"best_pattern": asdict(best_pattern)}
            )
            insights.append(insight)
        
        # 性能分化洞察
        if len(performance_data) >= 2:
            strategy_returns = {}
            for strategy_id, performances in performance_data.items():
                if performances:
                    avg_return = np.mean([p.return_rate for p in performances])
                    strategy_returns[strategy_id] = avg_return
            
            if strategy_returns:
                returns = list(strategy_returns.values())
                return_spread = max(returns) - min(returns)
                
                if return_spread > 0.05:  # 收益差距超过5%
                    insight = KnowledgeInsight(
                        insight_id="performance_divergence",
                        category="performance",
                        title="策略间存在显著性能差异",
                        description=f"策略间收益差距达到 {return_spread:.2%}，表明策略选择的重要性",
                        importance=0.7,
                        actionable=True,
                        supporting_evidence=[f"收益分布: {strategy_returns}"],
                        recommendations=[
                            "重点使用高收益策略",
                            "分析低收益策略的改进空间",
                            "建立动态策略选择机制"
                        ],
                        metadata={"return_spread": return_spread, "strategy_returns": strategy_returns}
                    )
                    insights.append(insight)
        
        return insights
    
    def _generate_optimization_insights(self, patterns: List[KnowledgePattern],
                                      rules: List[KnowledgeRule]) -> List[KnowledgeInsight]:
        """生成优化洞察"""
        insights = []
        
        # 模式覆盖洞察
        pattern_types = set(p.pattern_type for p in patterns)
        if len(pattern_types) < 3:
            insight = KnowledgeInsight(
                insight_id="limited_pattern_coverage",
                category="optimization",
                title="策略模式覆盖不足",
                description=f"当前仅识别出 {len(pattern_types)} 种模式类型，可能存在未发现的有效模式",
                importance=0.6,
                actionable=True,
                supporting_evidence=[f"模式类型: {list(pattern_types)}"],
                recommendations=[
                    "增加数据收集量",
                    "调整模式识别参数",
                    "尝试不同的分析方法"
                ],
                metadata={"pattern_types": list(pattern_types)}
            )
            insights.append(insight)
        
        # 规则质量洞察
        if rules:
            avg_rule_strength = np.mean([r.strength for r in rules])
            high_quality_rules = [r for r in rules if r.strength >= 0.7]
            
            insight = KnowledgeInsight(
                insight_id="rule_quality_assessment",
                category="optimization",
                title="知识规则质量评估",
                description=f"提取了 {len(rules)} 条规则，其中 {len(high_quality_rules)} 条为高质量规则 (强度 >= 0.7)",
                importance=0.5,
                actionable=True,
                supporting_evidence=[f"平均规则强度: {avg_rule_strength:.3f}"],
                recommendations=[
                    "重点应用高质量规则",
                    "验证和优化中等质量规则",
                    "持续更新和完善规则库"
                ],
                metadata={"total_rules": len(rules), "high_quality_rules": len(high_quality_rules)}
            )
            insights.append(insight)
        
        return insights
    
    def _generate_risk_insights(self, patterns: List[KnowledgePattern],
                              performance_data: Dict[str, List[StrategyPerformance]]) -> List[KnowledgeInsight]:
        """生成风险洞察"""
        insights = []
        
        # 相关性风险洞察
        correlation_patterns = [p for p in patterns if p.pattern_type == "correlation"]
        high_correlation_pairs = []
        
        for pattern in correlation_patterns:
            if "high_positive_corr" in pattern.pattern_id:
                high_correlation_pairs.append(pattern)
        
        if len(high_correlation_pairs) > 0:
            insight = KnowledgeInsight(
                insight_id="correlation_risk",
                category="risk_management",
                title="检测到高相关性风险",
                description=f"发现 {len(high_correlation_pairs)} 对高度正相关的策略，可能存在系统性风险",
                importance=0.8,
                actionable=True,
                supporting_evidence=[p.pattern_id for p in high_correlation_pairs],
                recommendations=[
                    "监控相关性变化",
                    "考虑增加负相关策略",
                    "设置相关性预警阈值"
                ],
                metadata={"high_corr_pairs": len(high_correlation_pairs)}
            )
            insights.append(insight)
        
        # 波动性风险洞察
        volatility_risks = []
        for strategy_id, performances in performance_data.items():
            if len(performances) >= 5:
                returns = [p.return_rate for p in performances]
                volatility = np.std(returns)
                if volatility > 0.1:  # 高波动性
                    volatility_risks.append((strategy_id, volatility))
        
        if volatility_risks:
            insight = KnowledgeInsight(
                insight_id="volatility_risk",
                category="risk_management",
                title="识别出高波动性策略",
                description=f"发现 {len(volatility_risks)} 个策略的波动性超过10%，需要重点关注风险控制",
                importance=0.7,
                actionable=True,
                supporting_evidence=[f"高波动策略: {volatility_risks}"],
                recommendations=[
                    "对高波动策略设置更严格的止损",
                    "考虑降低其权重",
                    "加强风险监控"
                ],
                metadata={"high_volatility_strategies": volatility_risks}
            )
            insights.append(insight)
        
        return insights
    
    def _generate_market_insights(self, strategies: List[BaseStrategy],
                                patterns: List[KnowledgePattern]) -> List[KnowledgeInsight]:
        """生成市场洞察"""
        insights = []
        
        # 策略适应性洞察
        rl_strategies = [s for s in strategies if s.strategy_type == StrategyType.REINFORCEMENT]
        if rl_strategies:
            insight = KnowledgeInsight(
                insight_id="adaptive_strategies",
                category="market_analysis",
                title="策略具有良好的市场适应性",
                description=f"当前组合包含 {len(rl_strategies)} 个强化学习策略，能够自适应市场变化",
                importance=0.6,
                actionable=True,
                supporting_evidence=[f"强化学习策略数量: {len(rl_strategies)}"],
                recommendations=[
                    "继续强化学习策略的训练",
                    "监控其学习效果",
                    "适时调整学习参数"
                ],
                metadata={"rl_strategy_count": len(rl_strategies)}
            )
            insights.append(insight)
        
        # 策略演进洞察
        evolution_strategies = [s for s in strategies if s.strategy_type == StrategyType.EVOLUTION]
        if evolution_strategies:
            insight = KnowledgeInsight(
                insight_id="evolutionary_advantage",
                category="market_analysis",
                title="策略组合具备进化优势",
                description=f"包含 {len(evolution_strategies)} 个进化算法策略，能够通过自然选择持续优化",
                importance=0.6,
                actionable=True,
                supporting_evidence=[f"进化策略数量: {len(evolution_strategies)}"],
                recommendations=[
                    "定期运行进化算法",
                    "监控种群多样性",
                    "调整进化参数"
                ],
                metadata={"evolution_strategy_count": len(evolution_strategies)}
            )
            insights.append(insight)
        
        return insights

class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.edge_counter = 0
    
    def build_knowledge_graph(self, strategies: List[BaseStrategy],
                            patterns: List[KnowledgePattern],
                            rules: List[KnowledgeRule],
                            insights: List[KnowledgeInsight]) -> nx.DiGraph:
        """构建知识图谱"""
        try:
            # 清空现有图
            self.graph.clear()
            self.node_counter = 0
            self.edge_counter = 0
            
            # 添加策略节点
            strategy_nodes = self._add_strategy_nodes(strategies)
            
            # 添加模式节点
            pattern_nodes = self._add_pattern_nodes(patterns)
            
            # 添加规则节点
            rule_nodes = self._add_rule_nodes(rules)
            
            # 添加洞察节点
            insight_nodes = self._add_insight_nodes(insights)
            
            # 添加边
            self._add_edges(strategies, patterns, rules, insights)
            
            logger.info(f"构建知识图谱完成: {self.graph.number_of_nodes()} 个节点, {self.graph.number_of_edges()} 条边")
            
            return self.graph
            
        except Exception as e:
            logger.error(f"构建知识图谱出错: {e}")
            return self.graph
    
    def _add_strategy_nodes(self, strategies: List[BaseStrategy]) -> Dict[str, str]:
        """添加策略节点"""
        strategy_nodes = {}
        
        for strategy in strategies:
            node_id = f"strategy_{strategy.strategy_id}"
            strategy_nodes[strategy.strategy_id] = node_id
            
            self.graph.add_node(
                node_id,
                node_type="strategy",
                label=f"策略: {strategy.strategy_id}",
                properties={
                    "strategy_id": strategy.strategy_id,
                    "strategy_type": strategy.strategy_type.value,
                    "usage_count": strategy.state.usage_count,
                    "success_rate": strategy.state.success_rate,
                    "is_active": strategy.is_active
                }
            )
            self.node_counter += 1
        
        return strategy_nodes
    
    def _add_pattern_nodes(self, patterns: List[KnowledgePattern]) -> Dict[str, str]:
        """添加模式节点"""
        pattern_nodes = {}
        
        for pattern in patterns:
            node_id = f"pattern_{pattern.pattern_id}"
            pattern_nodes[pattern.pattern_id] = node_id
            
            self.graph.add_node(
                node_id,
                node_type="pattern",
                label=f"模式: {pattern.pattern_type}",
                properties={
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "description": pattern.description,
                    "confidence": pattern.confidence,
                    "support_count": pattern.support_count
                }
            )
            self.node_counter += 1
        
        return pattern_nodes
    
    def _add_rule_nodes(self, rules: List[KnowledgeRule]) -> Dict[str, str]:
        """添加规则节点"""
        rule_nodes = {}
        
        for rule in rules:
            node_id = f"rule_{rule.rule_id}"
            rule_nodes[rule.rule_id] = node_id
            
            self.graph.add_node(
                node_id,
                node_type="rule",
                label=f"规则: {rule.rule_type}",
                properties={
                    "rule_id": rule.rule_id,
                    "rule_type": rule.rule_type,
                    "condition": rule.condition,
                    "conclusion": rule.conclusion,
                    "confidence": rule.confidence,
                    "strength": rule.strength
                }
            )
            self.node_counter += 1
        
        return rule_nodes
    
    def _add_insight_nodes(self, insights: List[KnowledgeInsight]) -> Dict[str, str]:
        """添加洞察节点"""
        insight_nodes = {}
        
        for insight in insights:
            node_id = f"insight_{insight.insight_id}"
            insight_nodes[insight.insight_id] = node_id
            
            self.graph.add_node(
                node_id,
                node_type="insight",
                label=f"洞察: {insight.title}",
                properties={
                    "insight_id": insight.insight_id,
                    "category": insight.category,
                    "title": insight.title,
                    "description": insight.description,
                    "importance": insight.importance,
                    "actionable": insight.actionable
                }
            )
            self.node_counter += 1
        
        return insight_nodes
    
    def _add_edges(self, strategies: List[BaseStrategy],
                  patterns: List[KnowledgePattern],
                  rules: List[KnowledgeRule],
                  insights: List[KnowledgeInsight]):
        """添加边"""
        # 策略-模式边
        for strategy in strategies:
            strategy_node = f"strategy_{strategy.strategy_id}"
            
            # 找到相关的模式
            related_patterns = []
            for pattern in patterns:
                if any(strategy.strategy_id in str(ex) for ex in pattern.examples):
                    related_patterns.append(pattern)
            
            for pattern in related_patterns:
                pattern_node = f"pattern_{pattern.pattern_id}"
                self.graph.add_edge(
                    strategy_node, pattern_node,
                    relationship_type="exhibits",
                    strength=pattern.confidence
                )
                self.edge_counter += 1
        
        # 模式-规则边
        for rule in rules:
            rule_node = f"rule_{rule.rule_id}"
            
            for pattern_id in rule.source_patterns:
                pattern_node = f"pattern_{pattern_id}"
                if self.graph.has_node(pattern_node):
                    self.graph.add_edge(
                        pattern_node, rule_node,
                        relationship_type="supports",
                        strength=rule.confidence
                    )
                    self.edge_counter += 1
        
        # 规则-洞察边
        for insight in insights:
            insight_node = f"insight_{insight.insight_id}"
            
            # 找到相关的规则
            for rule in rules:
                if rule.rule_type in insight.description.lower():
                    rule_node = f"rule_{rule.rule_id}"
                    if self.graph.has_node(rule_node):
                        self.graph.add_edge(
                            rule_node, insight_node,
                            relationship_type="leads_to",
                            strength=insight.importance
                        )
                        self.edge_counter += 1

class KnowledgeBase:
    """知识库管理"""
    
    def __init__(self):
        self.patterns = []
        self.rules = []
        self.insights = []
        self.graph = None
        self.knowledge_metadata = {}
        self.validation_results = {}
        
    def build_knowledge_base(self, strategies: List[BaseStrategy],
                           performance_data: Dict[str, List[StrategyPerformance]],
                           learning_contexts: List[LearningContext] = None) -> Dict[str, Any]:
        """构建完整知识库"""
        try:
            logger.info("开始构建知识库...")
            
            # 1. 模式识别
            pattern_recognizer = PatternRecognizer()
            self.patterns = pattern_recognizer.recognize_patterns(strategies, performance_data)
            
            # 2. 规则提取
            rule_extractor = RuleExtractor()
            self.rules = rule_extractor.extract_rules(self.patterns)
            
            # 3. 洞察生成
            insight_generator = InsightGenerator()
            self.insights = insight_generator.generate_insights(
                self.patterns, self.rules, strategies, performance_data
            )
            
            # 4. 知识图谱构建
            graph_builder = KnowledgeGraphBuilder()
            self.graph = graph_builder.build_knowledge_graph(
                strategies, self.patterns, self.rules, self.insights
            )
            
            # 5. 知识验证
            self.validation_results = self._validate_knowledge()
            
            # 6. 更新元数据
            self.knowledge_metadata = {
                'build_timestamp': datetime.now(),
                'patterns_count': len(self.patterns),
                'rules_count': len(self.rules),
                'insights_count': len(self.insights),
                'graph_nodes': self.graph.number_of_nodes() if self.graph else 0,
                'graph_edges': self.graph.number_of_edges() if self.graph else 0,
                'validation_score': np.mean([v.get('score', 0) for v in self.validation_results.values()]) if self.validation_results else 0
            }
            
            logger.info("知识库构建完成")
            
            return {
                'patterns': [asdict(p) for p in self.patterns],
                'rules': [asdict(r) for r in self.rules],
                'insights': [asdict(i) for i in self.insights],
                'graph_stats': {
                    'nodes': self.graph.number_of_nodes(),
                    'edges': self.graph.number_of_edges()
                },
                'metadata': self.knowledge_metadata,
                'validation_results': self.validation_results
            }
            
        except Exception as e:
            logger.error(f"构建知识库出错: {e}")
            return {'error': str(e)}
    
    def query_knowledge(self, query_type: str, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查询知识库"""
        try:
            if query_type == "patterns":
                return self._query_patterns(query_params)
            elif query_type == "rules":
                return self._query_rules(query_params)
            elif query_type == "insights":
                return self._query_insights(query_params)
            elif query_type == "graph":
                return self._query_graph(query_params)
            else:
                return []
                
        except Exception as e:
            logger.error(f"知识库查询出错: {e}")
            return []
    
    def _query_patterns(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查询模式"""
        filtered_patterns = self.patterns
        
        # 按类型过滤
        if 'pattern_type' in params:
            filtered_patterns = [p for p in filtered_patterns if p.pattern_type == params['pattern_type']]
        
        # 按置信度过滤
        if 'min_confidence' in params:
            filtered_patterns = [p for p in filtered_patterns if p.confidence >= params['min_confidence']]
        
        # 按支持度过滤
        if 'min_support' in params:
            filtered_patterns = [p for p in filtered_patterns if p.support_count >= params['min_support']]
        
        return [asdict(p) for p in filtered_patterns]
    
    def _query_rules(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查询规则"""
        filtered_rules = self.rules
        
        # 按类型过滤
        if 'rule_type' in params:
            filtered_rules = [r for r in filtered_rules if r.rule_type == params['rule_type']]
        
        # 按置信度过滤
        if 'min_confidence' in params:
            filtered_rules = [r for r in filtered_rules if r.confidence >= params['min_confidence']]
        
        # 按强度过滤
        if 'min_strength' in params:
            filtered_rules = [r for r in filtered_rules if r.strength >= params['min_strength']]
        
        return [asdict(r) for r in filtered_rules]
    
    def _query_insights(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查询洞察"""
        filtered_insights = self.insights
        
        # 按类别过滤
        if 'category' in params:
            filtered_insights = [i for i in filtered_insights if i.category == params['category']]
        
        # 按重要性过滤
        if 'min_importance' in params:
            filtered_insights = [i for i in filtered_insights if i.importance >= params['min_importance']]
        
        # 按可操作性过滤
        if 'actionable' in params:
            filtered_insights = [i for i in filtered_insights if i.actionable == params['actionable']]
        
        return [asdict(i) for i in filtered_insights]
    
    def _query_graph(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查询图谱"""
        if not self.graph:
            return []
        
        results = []
        
        # 按节点类型查询
        if 'node_type' in params:
            nodes = [(node, data) for node, data in self.graph.nodes(data=True) 
                    if data.get('node_type') == params['node_type']]
            results.extend([{'node': node, 'data': data} for node, data in nodes])
        
        # 按关系类型查询
        if 'relationship_type' in params:
            edges = [(u, v, data) for u, v, data in self.graph.edges(data=True)
                    if data.get('relationship_type') == params['relationship_type']]
            results.extend([{'source': u, 'target': v, 'data': data} for u, v, data in edges])
        
        # 路径查询
        if 'path_query' in params:
            source = params['path_query'].get('source')
            target = params['path_query'].get('target')
            if source and target and self.graph.has_node(source) and self.graph.has_node(target):
                try:
                    paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=3))
                    results.extend([{'path': path} for path in paths])
                except nx.NetworkXNoPath:
                    pass
        
        return results
    
    def _validate_knowledge(self) -> Dict[str, Any]:
        """验证知识质量"""
        validation_results = {}
        
        # 模式验证
        pattern_validation = self._validate_patterns()
        validation_results['patterns'] = pattern_validation
        
        # 规则验证
        rule_validation = self._validate_rules()
        validation_results['rules'] = rule_validation
        
        # 洞察验证
        insight_validation = self._validate_insights()
        validation_results['insights'] = insight_validation
        
        # 图谱验证
        graph_validation = self._validate_graph()
        validation_results['graph'] = graph_validation
        
        return validation_results
    
    def _validate_patterns(self) -> Dict[str, Any]:
        """验证模式质量"""
        if not self.patterns:
            return {'score': 0.0, 'issues': ['没有模式数据']}
        
        issues = []
        score = 1.0
        
        # 检查置信度分布
        confidences = [p.confidence for p in self.patterns]
        low_confidence_count = sum(1 for c in confidences if c < 0.5)
        if low_confidence_count > len(confidences) * 0.3:
            issues.append(f"低置信度模式过多: {low_confidence_count}/{len(confidences)}")
            score -= 0.2
        
        # 检查支持度分布
        supports = [p.support_count for p in self.patterns]
        low_support_count = sum(1 for s in supports if s < 3)
        if low_support_count > len(supports) * 0.4:
            issues.append(f"低支持度模式过多: {low_support_count}/{len(supports)}")
            score -= 0.1
        
        # 检查模式多样性
        pattern_types = set(p.pattern_type for p in self.patterns)
        if len(pattern_types) < 3:
            issues.append(f"模式类型多样性不足: {len(pattern_types)} 种类型")
            score -= 0.1
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'total_patterns': len(self.patterns),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'avg_support': np.mean(supports) if supports else 0.0
        }
    
    def _validate_rules(self) -> Dict[str, Any]:
        """验证规则质量"""
        if not self.rules:
            return {'score': 0.0, 'issues': ['没有规则数据']}
        
        issues = []
        score = 1.0
        
        # 检查规则强度分布
        strengths = [r.strength for r in self.rules]
        weak_rules_count = sum(1 for s in strengths if s < 0.5)
        if weak_rules_count > len(strengths) * 0.3:
            issues.append(f"弱规则过多: {weak_rules_count}/{len(strengths)}")
            score -= 0.2
        
        # 检查规则一致性
        rule_types = set(r.rule_type for r in self.rules)
        if len(rule_types) < 2:
            issues.append(f"规则类型单一: {len(rule_types)} 种类型")
            score -= 0.1
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'total_rules': len(self.rules),
            'avg_strength': np.mean(strengths) if strengths else 0.0
        }
    
    def _validate_insights(self) -> Dict[str, Any]:
        """验证洞察质量"""
        if not self.insights:
            return {'score': 0.0, 'issues': ['没有洞察数据']}
        
        issues = []
        score = 1.0
        
        # 检查洞察重要性分布
        importances = [i.importance for i in self.insights]
        low_importance_count = sum(1 for imp in importances if imp < 0.5)
        if low_importance_count > len(importances) * 0.4:
            issues.append(f"低重要性洞察过多: {low_importance_count}/{len(importances)}")
            score -= 0.1
        
        # 检查可操作性
        actionable_count = sum(1 for i in self.insights if i.actionable)
        if actionable_count < len(self.insights) * 0.6:
            issues.append(f"可操作性洞察不足: {actionable_count}/{len(self.insights)}")
            score -= 0.1
        
        # 检查类别多样性
        categories = set(i.category for i in self.insights)
        if len(categories) < 3:
            issues.append(f"洞察类别多样性不足: {len(categories)} 种类别")
            score -= 0.1
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'total_insights': len(self.insights),
            'avg_importance': np.mean(importances) if importances else 0.0,
            'actionable_ratio': actionable_count / len(self.insights) if self.insights else 0.0
        }
    
    def _validate_graph(self) -> Dict[str, Any]:
        """验证图谱质量"""
        if not self.graph:
            return {'score': 0.0, 'issues': ['没有图谱数据']}
        
        issues = []
        score = 1.0
        
        # 检查连通性
        if not nx.is_weakly_connected(self.graph):
            issues.append("图谱不连通")
            score -= 0.2
        
        # 检查孤立节点
        isolated_nodes = list(nx.isolates(self.graph))
        if isolated_nodes:
            issues.append(f"存在孤立节点: {len(isolated_nodes)} 个")
            score -= 0.1
        
        # 检查节点度分布
        degrees = [d for n, d in self.graph.degree()]
        if degrees:
            avg_degree = np.mean(degrees)
            if avg_degree < 2:
                issues.append(f"平均节点度偏低: {avg_degree:.2f}")
                score -= 0.1
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'isolated_nodes': len(isolated_nodes),
            'avg_degree': np.mean(degrees) if degrees else 0.0
        }
    
    def export_knowledge(self, filepath: str, format: str = 'json') -> bool:
        """导出知识库"""
        try:
            export_data = {
                'patterns': [asdict(p) for p in self.patterns],
                'rules': [asdict(r) for r in self.rules],
                'insights': [asdict(i) for i in self.insights],
                'metadata': self.knowledge_metadata,
                'validation_results': self.validation_results,
                'export_timestamp': datetime.now().isoformat()
            }
            
            if format == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logger.info(f"知识库已导出到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"导出知识库出错: {e}")
            return False
    
    def import_knowledge(self, filepath: str, format: str = 'json') -> bool:
        """导入知识库"""
        try:
            if format == 'json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
            else:
                raise ValueError(f"不支持的导入格式: {format}")
            
            # 重建对象
            self.patterns = [KnowledgePattern(**p) for p in import_data.get('patterns', [])]
            self.rules = [KnowledgeRule(**r) for r in import_data.get('rules', [])]
            self.insights = [KnowledgeInsight(**i) for i in import_data.get('insights', [])]
            self.knowledge_metadata = import_data.get('metadata', {})
            self.validation_results = import_data.get('validation_results', {})
            
            logger.info(f"知识库已从 {filepath} 导入")
            return True
            
        except Exception as e:
            logger.error(f"导入知识库出错: {e}")
            return False
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """获取知识库摘要"""
        return {
            'patterns_summary': {
                'total_count': len(self.patterns),
                'types': list(set(p.pattern_type for p in self.patterns)),
                'avg_confidence': np.mean([p.confidence for p in self.patterns]) if self.patterns else 0.0
            },
            'rules_summary': {
                'total_count': len(self.rules),
                'types': list(set(r.rule_type for r in self.rules)),
                'avg_strength': np.mean([r.strength for r in self.rules]) if self.rules else 0.0
            },
            'insights_summary': {
                'total_count': len(self.insights),
                'categories': list(set(i.category for i in self.insights)),
                'avg_importance': np.mean([i.importance for i in self.insights]) if self.insights else 0.0,
                'actionable_ratio': sum(1 for i in self.insights if i.actionable) / len(self.insights) if self.insights else 0.0
            },
            'graph_summary': {
                'nodes': self.graph.number_of_nodes() if self.graph else 0,
                'edges': self.graph.number_of_edges() if self.graph else 0,
                'is_connected': nx.is_weakly_connected(self.graph) if self.graph else False
            },
            'overall_quality': self.knowledge_metadata.get('validation_score', 0.0)
        }

# 工厂函数
def create_knowledge_extractor(extraction_method: str = 'comprehensive') -> KnowledgeBase:
    """创建知识提取器"""
    return KnowledgeBase()