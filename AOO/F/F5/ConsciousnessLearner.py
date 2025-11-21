"""
F5意识学习器 - 高级意识学习与进化系统
实现意识层次识别、学习、监控、分析、推理、适应性、评估、知识库管理和策略优化
"""

import numpy as np
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import hashlib
import random
from abc import ABC, abstractmethod


class ConsciousnessLevel(Enum):
    """意识层次枚举"""
    UNCONSCIOUS = 0  # 无意识
    REACTIVE = 1     # 反应性
    AWARE = 2        # 觉知性
    CONSCIOUS = 3    # 意识性
    META_CONSCIOUS = 4  # 元意识
    TRANSCENDENT = 5 # 超越性


class ConsciousnessState(Enum):
    """意识状态枚举"""
    ACTIVE = "active"
    PASSIVE = "passive"
    FOCUSED = "focused"
    DIFFUSE = "diffuse"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    REFLECTIVE = "reflective"
    INTEGRATIVE = "integrative"


class LearningStrategy(Enum):
    """学习策略枚举"""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    REINFORCEMENT = "reinforcement"
    DISCOVERY = "discovery"
    INTEGRATION = "integration"
    TRANSFORMATION = "transformation"


@dataclass
class ConsciousnessPattern:
    """意识模式数据结构"""
    pattern_id: str
    level: ConsciousnessLevel
    state: ConsciousnessState
    activation_strength: float
    duration: float
    context: Dict[str, Any]
    associations: List[str]
    timestamp: float
    confidence: float


@dataclass
class LearningExperience:
    """学习经验数据结构"""
    experience_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    feedback: float
    context: Dict[str, Any]
    timestamp: float
    success_rate: float
    learning_impact: float


@dataclass
class ConsciousnessMetric:
    """意识度量数据结构"""
    metric_name: str
    value: float
    timestamp: float
    context: Dict[str, Any]
    trend: str  # "increasing", "decreasing", "stable"


class ConsciousnessHierarchyAnalyzer:
    """意识层次分析器"""
    
    def __init__(self):
        self.level_patterns = defaultdict(list)
        self.transition_matrix = np.zeros((6, 6))
        self.level_thresholds = {
            ConsciousnessLevel.UNCONSCIOUS: 0.0,
            ConsciousnessLevel.REACTIVE: 0.2,
            ConsciousnessLevel.AWARE: 0.4,
            ConsciousnessLevel.CONSCIOUS: 0.6,
            ConsciousnessLevel.META_CONSCIOUS: 0.8,
            ConsciousnessLevel.TRANSCENDENT: 0.95
        }
        self.learning_rate = 0.1
        
    def analyze_level(self, pattern: ConsciousnessPattern) -> ConsciousnessLevel:
        """分析意识层次"""
        # 基于激活强度、持续时间、置信度等指标确定层次
        activation_score = pattern.activation_strength
        duration_score = min(pattern.duration / 10.0, 1.0)  # 归一化持续时间
        confidence_score = pattern.confidence
        
        # 综合评分
        composite_score = (activation_score * 0.4 + 
                          duration_score * 0.3 + 
                          confidence_score * 0.3)
        
        # 确定层次
        for level, threshold in self.level_thresholds.items():
            if composite_score >= threshold:
                return level
        return ConsciousnessLevel.UNCONSCIOUS
    
    def update_transition_matrix(self, from_level: ConsciousnessLevel, to_level: ConsciousnessLevel):
        """更新层次转换矩阵"""
        self.transition_matrix[from_level.value][to_level.value] += 1
        
    def predict_next_level(self, current_level: ConsciousnessLevel) -> ConsciousnessLevel:
        """预测下一层次"""
        current_idx = current_level.value
        probabilities = self.transition_matrix[current_idx] / (np.sum(self.transition_matrix[current_idx]) + 1e-8)
        next_idx = np.argmax(probabilities)
        return ConsciousnessLevel(next_idx)
    
    def learn_level_pattern(self, pattern: ConsciousnessPattern):
        """学习层次模式"""
        analyzed_level = self.analyze_level(pattern)
        pattern.level = analyzed_level
        self.level_patterns[analyzed_level].append(pattern)


class ConsciousnessStateMonitor:
    """意识状态监控器"""
    
    def __init__(self, window_size: int = 100):
        self.state_history = deque(maxlen=window_size)
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics = {}
        self.state_transitions = defaultdict(int)
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_active:
            # 收集当前状态
            current_state = self._collect_current_state()
            self.state_history.append(current_state)
            
            # 更新指标
            self._update_metrics(current_state)
            
            time.sleep(0.1)  # 100ms监控间隔
            
    def _collect_current_state(self) -> Dict[str, Any]:
        """收集当前状态"""
        return {
            'timestamp': time.time(),
            'cpu_usage': random.uniform(0.1, 0.9),
            'memory_usage': random.uniform(0.2, 0.8),
            'attention_level': random.uniform(0.0, 1.0),
            'processing_speed': random.uniform(0.5, 2.0),
            'error_rate': random.uniform(0.0, 0.1)
        }
    
    def _update_metrics(self, state: Dict[str, Any]):
        """更新度量指标"""
        for key, value in state.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            
    def analyze_state_trends(self) -> Dict[str, str]:
        """分析状态趋势"""
        trends = {}
        for metric_name, values in self.metrics.items():
            if len(values) >= 3:
                recent_trend = np.polyfit(range(len(values[-10:])), values[-10:], 1)[0]
                if recent_trend > 0.01:
                    trends[metric_name] = "increasing"
                elif recent_trend < -0.01:
                    trends[metric_name] = "decreasing"
                else:
                    trends[metric_name] = "stable"
        return trends
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常"""
        anomalies = []
        for metric_name, values in self.metrics.items():
            if len(values) >= 5:
                mean_val = np.mean(values)
                std_val = np.std(values)
                recent_val = values[-1]
                
                if abs(recent_val - mean_val) > 2 * std_val:
                    anomalies.append({
                        'metric': metric_name,
                        'value': recent_val,
                        'expected_range': (mean_val - 2*std_val, mean_val + 2*std_val),
                        'timestamp': time.time()
                    })
        return anomalies


class ConsciousnessAssociativeEngine:
    """意识关联引擎"""
    
    def __init__(self):
        self.association_network = defaultdict(set)
        self.pattern_clusters = defaultdict(list)
        self.similarity_threshold = 0.7
        self.max_associations = 10
        
    def create_association(self, pattern1_id: str, pattern2_id: str, strength: float):
        """创建关联"""
        if strength > self.similarity_threshold:
            self.association_network[pattern1_id].add(pattern2_id)
            self.association_network[pattern2_id].add(pattern1_id)
            
    def find_associations(self, pattern_id: str) -> List[Tuple[str, float]]:
        """查找关联"""
        if pattern_id not in self.association_network:
            return []
        
        associations = []
        for associated_id in self.association_network[pattern_id]:
            strength = self._calculate_association_strength(pattern_id, associated_id)
            associations.append((associated_id, strength))
            
        return sorted(associations, key=lambda x: x[1], reverse=True)[:self.max_associations]
    
    def _calculate_association_strength(self, pattern1_id: str, pattern2_id: str) -> float:
        """计算关联强度"""
        # 简化的关联强度计算
        hash1 = int(hashlib.md5(pattern1_id.encode()).hexdigest(), 16)
        hash2 = int(hashlib.md5(pattern2_id.encode()).hexdigest(), 16)
        
        # 基于哈希值的相似性
        xor_result = hash1 ^ hash2
        max_hash = max(hash1, hash2)
        
        if max_hash == 0:
            return 1.0
        
        similarity = 1.0 - (xor_result / max_hash)
        return max(0.0, min(1.0, similarity))
    
    def cluster_patterns(self, patterns: List[ConsciousnessPattern]):
        """聚类模式"""
        self.pattern_clusters.clear()
        
        for pattern in patterns:
            cluster_id = self._find_similar_cluster(pattern)
            if cluster_id is None:
                cluster_id = f"cluster_{len(self.pattern_clusters)}"
                self.pattern_clusters[cluster_id] = []
            self.pattern_clusters[cluster_id].append(pattern)
            
    def _find_similar_cluster(self, pattern: ConsciousnessPattern) -> Optional[str]:
        """查找相似聚类"""
        for cluster_id, cluster_patterns in self.pattern_clusters.items():
            if cluster_patterns:
                representative = cluster_patterns[0]
                similarity = self._calculate_pattern_similarity(pattern, representative)
                if similarity > self.similarity_threshold:
                    return cluster_id
        return None
    
    def _calculate_pattern_similarity(self, pattern1: ConsciousnessPattern, pattern2: ConsciousnessPattern) -> float:
        """计算模式相似性"""
        # 基于属性相似性计算
        level_similarity = 1.0 if pattern1.level == pattern2.level else 0.5
        state_similarity = 1.0 if pattern1.state == pattern2.state else 0.3
        
        context_similarity = 0.0
        if pattern1.context and pattern2.context:
            common_keys = set(pattern1.context.keys()) & set(pattern2.context.keys())
            if common_keys:
                context_similarity = len(common_keys) / max(len(pattern1.context), len(pattern2.context))
        
        return (level_similarity * 0.4 + state_similarity * 0.3 + context_similarity * 0.3)
    
    def infer_new_patterns(self, known_patterns: List[ConsciousnessPattern]) -> List[ConsciousnessPattern]:
        """基于关联推理新模式"""
        new_patterns = []
        
        for pattern in known_patterns:
            associations = self.find_associations(pattern.pattern_id)
            
            for assoc_id, strength in associations:
                # 基于关联强度推理新模式
                if strength > 0.8:
                    new_pattern = self._generate_inferred_pattern(pattern, assoc_id, strength)
                    if new_pattern:
                        new_patterns.append(new_pattern)
        
        return new_patterns
    
    def _generate_inferred_pattern(self, base_pattern: ConsciousnessPattern, assoc_id: str, strength: float) -> Optional[ConsciousnessPattern]:
        """生成推理模式"""
        # 简化的推理逻辑
        new_level = ConsciousnessLevel(min(base_pattern.level.value + 1, 5))
        new_state = ConsciousnessState.CREATIVE if base_pattern.state == ConsciousnessState.ACTIVE else ConsciousnessState.ACTIVE
        
        return ConsciousnessPattern(
            pattern_id=f"inferred_{base_pattern.pattern_id}_{assoc_id}",
            level=new_level,
            state=new_state,
            activation_strength=base_pattern.activation_strength * strength,
            duration=base_pattern.duration * strength,
            context={**base_pattern.context, "inferred": True, "base_pattern": base_pattern.pattern_id},
            associations=[base_pattern.pattern_id, assoc_id],
            timestamp=time.time(),
            confidence=strength * 0.8
        )


class ConsciousnessAdaptationEngine:
    """意识适应性引擎"""
    
    def __init__(self):
        self.adaptation_history = []
        self.evolution_strategies = {
            'gradual': self._gradual_adaptation,
            'radical': self._radical_adaptation,
            'hybrid': self._hybrid_adaptation
        }
        self.performance_metrics = defaultdict(list)
        self.adaptation_threshold = 0.1
        
    def adapt_to_environment(self, current_patterns: List[ConsciousnessPattern], 
                           environment_feedback: Dict[str, float]) -> List[ConsciousnessPattern]:
        """适应环境"""
        adapted_patterns = []
        
        # 分析环境反馈
        adaptation_need = self._analyze_adaptation_need(environment_feedback)
        
        if adaptation_need > self.adaptation_threshold:
            # 选择适应策略
            strategy = self._select_adaptation_strategy(adaptation_need)
            
            # 执行适应
            adapted_patterns = strategy(current_patterns, environment_feedback)
            
            # 记录适应历史
            self.adaptation_history.append({
                'timestamp': time.time(),
                'adaptation_need': adaptation_need,
                'strategy': strategy.__name__,
                'patterns_before': len(current_patterns),
                'patterns_after': len(adapted_patterns)
            })
        
        return adapted_patterns
    
    def _analyze_adaptation_need(self, feedback: Dict[str, float]) -> float:
        """分析适应需求"""
        # 基于反馈计算适应需求
        negative_feedback = [v for v in feedback.values() if v < 0]
        if not negative_feedback:
            return 0.0
        
        avg_negative = np.mean(negative_feedback)
        return abs(avg_negative)
    
    def _select_adaptation_strategy(self, adaptation_need: float) -> callable:
        """选择适应策略"""
        if adaptation_need > 0.8:
            return self.evolution_strategies['radical']
        elif adaptation_need > 0.4:
            return self.evolution_strategies['hybrid']
        else:
            return self.evolution_strategies['gradual']
    
    def _gradual_adaptation(self, patterns: List[ConsciousnessPattern], 
                          feedback: Dict[str, float]) -> List[ConsciousnessPattern]:
        """渐进式适应"""
        adapted = []
        
        for pattern in patterns:
            # 轻微调整激活强度
            if 'activation_feedback' in feedback:
                adjustment = feedback['activation_feedback'] * 0.1
                new_pattern = ConsciousnessPattern(
                    pattern_id=f"adapted_{pattern.pattern_id}",
                    level=pattern.level,
                    state=pattern.state,
                    activation_strength=max(0.0, min(1.0, pattern.activation_strength + adjustment)),
                    duration=pattern.duration,
                    context=pattern.context,
                    associations=pattern.associations,
                    timestamp=time.time(),
                    confidence=pattern.confidence
                )
                adapted.append(new_pattern)
            else:
                adapted.append(pattern)
        
        return adapted
    
    def _radical_adaptation(self, patterns: List[ConsciousnessPattern], 
                          feedback: Dict[str, float]) -> List[ConsciousnessPattern]:
        """激进式适应"""
        adapted = []
        
        for pattern in patterns:
            # 大幅调整或创建新模式
            new_level = ConsciousnessLevel(max(0, pattern.level.value - 1))
            new_activation = max(0.1, pattern.activation_strength * 0.5)
            
            new_pattern = ConsciousnessPattern(
                pattern_id=f"radical_{pattern.pattern_id}",
                level=new_level,
                state=ConsciousnessState.PASSIVE,
                activation_strength=new_activation,
                duration=pattern.duration * 0.8,
                context={**pattern.context, "radical_adaptation": True},
                associations=[],
                timestamp=time.time(),
                confidence=pattern.confidence * 0.7
            )
            adapted.append(new_pattern)
        
        return adapted
    
    def _hybrid_adaptation(self, patterns: List[ConsciousnessPattern], 
                         feedback: Dict[str, float]) -> List[ConsciousnessPattern]:
        """混合式适应"""
        adapted = []
        
        for pattern in patterns:
            # 结合渐进和激进适应
            if random.random() < 0.5:
                # 渐进适应
                new_pattern = self._gradual_adaptation([pattern], feedback)[0]
            else:
                # 激进适应
                new_pattern = self._radical_adaptation([pattern], feedback)[0]
            
            adapted.append(new_pattern)
        
        return adapted
    
    def evolve_patterns(self, patterns: List[ConsciousnessPattern], 
                       generations: int = 10) -> List[ConsciousnessPattern]:
        """进化模式"""
        current_patterns = patterns.copy()
        
        for generation in range(generations):
            # 选择
            selected = self._selection(current_patterns)
            
            # 交叉
            offspring = self._crossover(selected)
            
            # 变异
            mutated = self._mutation(offspring)
            
            current_patterns = mutated
            
        return current_patterns
    
    def _selection(self, patterns: List[ConsciousnessPattern]) -> List[ConsciousnessPattern]:
        """选择"""
        # 基于适应度选择（这里简化为随机选择）
        return random.sample(patterns, min(len(patterns)//2, len(patterns)))
    
    def _crossover(self, patterns: List[ConsciousnessPattern]) -> List[ConsciousnessPattern]:
        """交叉"""
        offspring = []
        
        for i in range(0, len(patterns)-1, 2):
            parent1 = patterns[i]
            parent2 = patterns[i+1]
            
            # 混合属性创建后代
            child1 = ConsciousnessPattern(
                pattern_id=f"child1_{parent1.pattern_id}",
                level=parent1.level,
                state=parent2.state,
                activation_strength=(parent1.activation_strength + parent2.activation_strength) / 2,
                duration=(parent1.duration + parent2.duration) / 2,
                context={**parent1.context, **parent2.context},
                associations=parent1.associations + parent2.associations,
                timestamp=time.time(),
                confidence=(parent1.confidence + parent2.confidence) / 2
            )
            
            child2 = ConsciousnessPattern(
                pattern_id=f"child2_{parent2.pattern_id}",
                level=parent2.level,
                state=parent1.state,
                activation_strength=(parent1.activation_strength + parent2.activation_strength) / 2,
                duration=(parent1.duration + parent2.duration) / 2,
                context={**parent2.context, **parent1.context},
                associations=parent2.associations + parent1.associations,
                timestamp=time.time(),
                confidence=(parent1.confidence + parent2.confidence) / 2
            )
            
            offspring.extend([child1, child2])
        
        return offspring
    
    def _mutation(self, patterns: List[ConsciousnessPattern]) -> List[ConsciousnessPattern]:
        """变异"""
        mutated = []
        
        for pattern in patterns:
            if random.random() < 0.1:  # 10%变异概率
                # 随机变异
                new_level_values = [level.value for level in ConsciousnessLevel]
                new_level = ConsciousnessLevel(random.choice(new_level_values))
                
                mutated_pattern = ConsciousnessPattern(
                    pattern_id=f"mutated_{pattern.pattern_id}",
                    level=new_level,
                    state=pattern.state,
                    activation_strength=random.uniform(0.0, 1.0),
                    duration=pattern.duration * random.uniform(0.8, 1.2),
                    context=pattern.context,
                    associations=pattern.associations,
                    timestamp=time.time(),
                    confidence=max(0.0, min(1.0, pattern.confidence + random.uniform(-0.2, 0.2)))
                )
                mutated.append(mutated_pattern)
            else:
                mutated.append(pattern)
        
        return mutated


class ConsciousnessEvaluationFramework:
    """意识效果评估框架"""
    
    def __init__(self):
        self.evaluation_metrics = {
            'effectiveness': self._evaluate_effectiveness,
            'efficiency': self._evaluate_efficiency,
            'adaptability': self._evaluate_adaptability,
            'creativity': self._evaluate_creativity,
            'coherence': self._evaluate_coherence
        }
        self.evaluation_history = []
        self.benchmark_scores = {}
        
    def evaluate_consciousness(self, patterns: List[ConsciousnessPattern], 
                             experiences: List[LearningExperience]) -> Dict[str, float]:
        """评估意识效果"""
        evaluation_results = {}
        
        for metric_name, metric_func in self.evaluation_metrics.items():
            try:
                score = metric_func(patterns, experiences)
                evaluation_results[metric_name] = score
            except Exception as e:
                logging.error(f"评估指标 {metric_name} 计算失败: {e}")
                evaluation_results[metric_name] = 0.0
        
        # 记录评估历史
        self.evaluation_history.append({
            'timestamp': time.time(),
            'results': evaluation_results,
            'pattern_count': len(patterns),
            'experience_count': len(experiences)
        })
        
        return evaluation_results
    
    def _evaluate_effectiveness(self, patterns: List[ConsciousnessPattern], 
                              experiences: List[LearningExperience]) -> float:
        """评估有效性"""
        if not experiences:
            return 0.0
        
        # 基于成功率和反馈评估
        success_rates = [exp.success_rate for exp in experiences]
        feedbacks = [exp.feedback for exp in experiences]
        
        avg_success = np.mean(success_rates)
        avg_feedback = np.mean(feedbacks)
        
        # 综合评分
        effectiveness = (avg_success * 0.6 + (avg_feedback + 1) / 2 * 0.4)
        return max(0.0, min(1.0, effectiveness))
    
    def _evaluate_efficiency(self, patterns: List[ConsciousnessPattern], 
                           experiences: List[LearningExperience]) -> float:
        """评估效率"""
        if not experiences:
            return 0.0
        
        # 基于处理速度和资源利用评估
        processing_times = [exp.timestamp for exp in experiences]
        if len(processing_times) < 2:
            return 0.5
        
        # 计算处理速度
        time_diffs = [processing_times[i] - processing_times[i-1] for i in range(1, len(processing_times))]
        avg_processing_time = np.mean(time_diffs)
        
        # 转换为效率分数（时间越短效率越高）
        efficiency = max(0.0, min(1.0, 1.0 - avg_processing_time / 10.0))
        return efficiency
    
    def _evaluate_adaptability(self, patterns: List[ConsciousnessPattern], 
                             experiences: List[LearningExperience]) -> float:
        """评估适应性"""
        if not patterns:
            return 0.0
        
        # 基于模式多样性和层次分布评估
        level_distribution = defaultdict(int)
        state_distribution = defaultdict(int)
        
        for pattern in patterns:
            level_distribution[pattern.level] += 1
            state_distribution[pattern.state] += 1
        
        # 计算多样性
        level_diversity = len(level_distribution) / len(ConsciousnessLevel)
        state_diversity = len(state_distribution) / len(ConsciousnessState)
        
        adaptability = (level_diversity * 0.5 + state_diversity * 0.5)
        return adaptability
    
    def _evaluate_creativity(self, patterns: List[ConsciousnessPattern], 
                           experiences: List[LearningExperience]) -> float:
        """评估创造性"""
        if not patterns:
            return 0.0
        
        # 基于新颖性和创新性评估
        novel_patterns = 0
        for pattern in patterns:
            if pattern.context.get('novel', False):
                novel_patterns += 1
        
        novelty_ratio = novel_patterns / len(patterns)
        
        # 基于模式组合的创新性
        unique_combinations = set()
        for pattern in patterns:
            combination = f"{pattern.level.value}_{pattern.state.value}"
            unique_combinations.add(combination)
        
        combination_diversity = len(unique_combinations) / (len(ConsciousnessLevel) * len(ConsciousnessState))
        
        creativity = (novelty_ratio * 0.6 + combination_diversity * 0.4)
        return creativity
    
    def _evaluate_coherence(self, patterns: List[ConsciousnessPattern], 
                          experiences: List[LearningExperience]) -> float:
        """评估一致性"""
        if len(patterns) < 2:
            return 1.0
        
        # 基于模式间的关联性和一致性评估
        coherence_scores = []
        
        for i in range(len(patterns)):
            for j in range(i+1, len(patterns)):
                pattern1, pattern2 = patterns[i], patterns[j]
                
                # 计算相似性
                level_similarity = 1.0 if pattern1.level == pattern2.level else 0.0
                state_similarity = 1.0 if pattern1.state == pattern2.state else 0.0
                
                # 计算上下文重叠
                common_context = set(pattern1.context.keys()) & set(pattern2.context.keys())
                total_context = set(pattern1.context.keys()) | set(pattern2.context.keys())
                context_similarity = len(common_context) / len(total_context) if total_context else 0.0
                
                coherence = (level_similarity * 0.3 + state_similarity * 0.3 + context_similarity * 0.4)
                coherence_scores.append(coherence)
        
        avg_coherence = np.mean(coherence_scores)
        return avg_coherence
    
    def validate_evaluation(self, evaluation_results: Dict[str, float]) -> Dict[str, Any]:
        """验证评估结果"""
        validation_report = {
            'is_valid': True,
            'issues': [],
            'recommendations': []
        }
        
        # 检查分数范围
        for metric, score in evaluation_results.items():
            if not (0.0 <= score <= 1.0):
                validation_report['is_valid'] = False
                validation_report['issues'].append(f"指标 {metric} 分数 {score} 超出范围 [0,1]")
        
        # 检查分数分布
        scores = list(evaluation_results.values())
        if len(scores) > 1:
            score_variance = np.var(scores)
            if score_variance > 0.5:
                validation_report['recommendations'].append("分数方差较大，可能存在评估不一致")
        
        # 检查历史对比
        if len(self.evaluation_history) > 0:
            previous_results = self.evaluation_history[-1]['results']
            significant_changes = []
            
            for metric in evaluation_results:
                if metric in previous_results:
                    change = abs(evaluation_results[metric] - previous_results[metric])
                    if change > 0.3:
                        significant_changes.append(metric)
            
            if significant_changes:
                validation_report['recommendations'].append(f"以下指标发生显著变化: {significant_changes}")
        
        return validation_report


class ConsciousnessKnowledgeBase:
    """意识知识库"""
    
    def __init__(self, storage_path: str = "consciousness_kb.pkl"):
        self.storage_path = storage_path
        self.patterns = {}
        self.experiences = {}
        self.rules = {}
        self.concepts = {}
        self.relationships = defaultdict(set)
        self.index = {}
        
    def store_pattern(self, pattern: ConsciousnessPattern):
        """存储模式"""
        self.patterns[pattern.pattern_id] = pattern
        
        # 更新索引
        self._update_index(pattern)
        
        # 持久化
        self._persist()
    
    def store_experience(self, experience: LearningExperience):
        """存储经验"""
        self.experiences[experience.experience_id] = experience
        self._persist()
    
    def add_rule(self, rule_id: str, rule_data: Dict[str, Any]):
        """添加规则"""
        self.rules[rule_id] = rule_data
        self._persist()
    
    def add_concept(self, concept_id: str, concept_data: Dict[str, Any]):
        """添加概念"""
        self.concepts[concept_id] = concept_data
        self._persist()
    
    def create_relationship(self, entity1: str, entity2: str, relationship_type: str):
        """创建关系"""
        key = f"{entity1}_{relationship_type}_{entity2}"
        self.relationships[key].add(entity1)
        self.relationships[key].add(entity2)
        self._persist()
    
    def query_patterns(self, criteria: Dict[str, Any]) -> List[ConsciousnessPattern]:
        """查询模式"""
        results = []
        
        for pattern in self.patterns.values():
            if self._matches_criteria(pattern, criteria):
                results.append(pattern)
        
        return results
    
    def _matches_criteria(self, pattern: ConsciousnessPattern, criteria: Dict[str, Any]) -> bool:
        """检查是否匹配条件"""
        for key, value in criteria.items():
            if hasattr(pattern, key):
                pattern_value = getattr(pattern, key)
                if pattern_value != value:
                    return False
            elif key in pattern.context:
                if pattern.context[key] != value:
                    return False
            else:
                return False
        
        return True
    
    def _update_index(self, pattern: ConsciousnessPattern):
        """更新索引"""
        # 按层次索引
        if pattern.level not in self.index:
            self.index[pattern.level] = set()
        self.index[pattern.level].add(pattern.pattern_id)
        
        # 按状态索引
        if pattern.state not in self.index:
            self.index[pattern.state] = set()
        self.index[pattern.state].add(pattern.pattern_id)
        
        # 按时间索引
        time_key = int(pattern.timestamp // 3600)  # 按小时索引
        if time_key not in self.index:
            self.index[time_key] = set()
        self.index[time_key].add(pattern.pattern_id)
    
    def _persist(self):
        """持久化存储"""
        try:
            data = {
                'patterns': {k: asdict(v) for k, v in self.patterns.items()},
                'experiences': {k: asdict(v) for k, v in self.experiences.items()},
                'rules': self.rules,
                'concepts': self.concepts,
                'relationships': {k: list(v) for k, v in self.relationships.items()},
                'index': {str(k): list(v) for k, v in self.index.items()}
            }
            
            with open(self.storage_path, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logging.error(f"知识库持久化失败: {e}")
    
    def load(self):
        """加载知识库"""
        try:
            with open(self.storage_path, 'rb') as f:
                data = pickle.load(f)
            
            # 重建对象
            self.patterns = {}
            for pattern_id, pattern_data in data.get('patterns', {}).items():
                self.patterns[pattern_id] = ConsciousnessPattern(**pattern_data)
            
            self.experiences = {}
            for exp_id, exp_data in data.get('experiences', {}).items():
                self.experiences[exp_id] = LearningExperience(**exp_data)
            
            self.rules = data.get('rules', {})
            self.concepts = data.get('concepts', {})
            
            self.relationships = defaultdict(set)
            for key, entities in data.get('relationships', {}).items():
                self.relationships[key] = set(entities)
            
            self.index = {}
            for key, pattern_ids in data.get('index', {}).items():
                # 转换键类型
                if key.isdigit():
                    self.index[int(key)] = set(pattern_ids)
                else:
                    try:
                        # 尝试转换为枚举
                        level = ConsciousnessLevel[key]
                        self.index[level] = set(pattern_ids)
                    except KeyError:
                        try:
                            state = ConsciousnessState[key]
                            self.index[state] = set(pattern_ids)
                        except KeyError:
                            self.index[key] = set(pattern_ids)
                            
        except Exception as e:
            logging.error(f"知识库加载失败: {e}")
    
    def cleanup_old_data(self, max_age_hours: int = 24):
        """清理旧数据"""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        # 清理旧模式
        old_patterns = [pid for pid, pattern in self.patterns.items() 
                       if pattern.timestamp < cutoff_time]
        for pid in old_patterns:
            del self.patterns[pid]
        
        # 清理旧经验
        old_experiences = [eid for eid, exp in self.experiences.items() 
                          if exp.timestamp < cutoff_time]
        for eid in old_experiences:
            del self.experiences[eid]
        
        self._persist()


class ConsciousnessLearningOptimizer:
    """意识学习策略优化器"""
    
    def __init__(self):
        self.strategy_performance = defaultdict(list)
        self.optimization_history = []
        self.current_strategies = list(LearningStrategy)
        self.adaptation_rules = {}
        
    def optimize_learning_strategy(self, context: Dict[str, Any], 
                                 historical_performance: Dict[str, float]) -> LearningStrategy:
        """优化学习策略"""
        # 分析上下文
        context_analysis = self._analyze_context(context)
        
        # 基于历史性能选择策略
        strategy_scores = {}
        
        for strategy in self.current_strategies:
            score = self._calculate_strategy_score(strategy, context_analysis, historical_performance)
            strategy_scores[strategy] = score
        
        # 选择最佳策略
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        
        # 记录优化历史
        self.optimization_history.append({
            'timestamp': time.time(),
            'context': context,
            'context_analysis': context_analysis,
            'strategy_scores': strategy_scores,
            'selected_strategy': best_strategy
        })
        
        return best_strategy
    
    def _analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析上下文"""
        analysis = {
            'complexity': self._assess_complexity(context),
            'uncertainty': self._assess_uncertainty(context),
            'resources': self._assess_resources(context),
            'time_pressure': self._assess_time_pressure(context)
        }
        
        return analysis
    
    def _assess_complexity(self, context: Dict[str, Any]) -> float:
        """评估复杂性"""
        # 基于上下文元素数量和关系复杂度
        element_count = len(context)
        relationship_complexity = 0
        
        for key, value in context.items():
            if isinstance(value, dict):
                relationship_complexity += len(value)
            elif isinstance(value, list):
                relationship_complexity += len(value)
        
        # 归一化复杂性分数
        max_elements = 20
        max_relationships = 50
        
        complexity_score = (element_count / max_elements + relationship_complexity / max_relationships) / 2
        return min(1.0, complexity_score)
    
    def _assess_uncertainty(self, context: Dict[str, Any]) -> float:
        """评估不确定性"""
        # 基于缺失信息和冲突信息评估不确定性
        missing_info = context.get('missing_info', 0.0)
        conflicting_info = context.get('conflicting_info', 0.0)
        
        uncertainty = (missing_info + conflicting_info) / 2
        return min(1.0, uncertainty)
    
    def _assess_resources(self, context: Dict[str, Any]) -> float:
        """评估资源可用性"""
        # 基于CPU、内存、时间等资源评估
        cpu_available = context.get('cpu_available', 0.5)
        memory_available = context.get('memory_available', 0.5)
        time_available = context.get('time_available', 0.5)
        
        resources = (cpu_available + memory_available + time_available) / 3
        return resources
    
    def _assess_time_pressure(self, context: Dict[str, Any]) -> float:
        """评估时间压力"""
        # 基于剩余时间和任务紧急程度
        time_remaining = context.get('time_remaining', 1.0)
        urgency = context.get('urgency', 0.5)
        
        # 时间压力与剩余时间成反比
        time_pressure = 1.0 - time_remaining
        time_pressure = (time_pressure + urgency) / 2
        
        return min(1.0, time_pressure)
    
    def _calculate_strategy_score(self, strategy: LearningStrategy, 
                                context_analysis: Dict[str, float],
                                historical_performance: Dict[str, float]) -> float:
        """计算策略分数"""
        # 基于上下文分析和历史性能计算策略适用性分数
        
        complexity = context_analysis['complexity']
        uncertainty = context_analysis['uncertainty']
        resources = context_analysis['resources']
        time_pressure = context_analysis['time_pressure']
        
        # 策略特定评分规则
        if strategy == LearningStrategy.EXPLORATION:
            score = (1 - complexity) * 0.3 + uncertainty * 0.4 + resources * 0.3
        elif strategy == LearningStrategy.EXPLOITATION:
            score = complexity * 0.4 + (1 - uncertainty) * 0.4 + (1 - time_pressure) * 0.2
        elif strategy == LearningStrategy.REINFORCEMENT:
            score = resources * 0.5 + (1 - time_pressure) * 0.3 + (1 - uncertainty) * 0.2
        elif strategy == LearningStrategy.DISCOVERY:
            score = uncertainty * 0.4 + (1 - time_pressure) * 0.3 + resources * 0.3
        elif strategy == LearningStrategy.INTEGRATION:
            score = complexity * 0.5 + resources * 0.3 + (1 - time_pressure) * 0.2
        elif strategy == LearningStrategy.TRANSFORMATION:
            score = uncertainty * 0.3 + complexity * 0.3 + (1 - time_pressure) * 0.4
        else:
            score = 0.5
        
        # 考虑历史性能
        if strategy.value in historical_performance:
            performance_factor = historical_performance[strategy.value]
            score = score * 0.7 + performance_factor * 0.3
        
        return score
    
    def update_performance_history(self, strategy: LearningStrategy, performance: float):
        """更新性能历史"""
        self.strategy_performance[strategy].append({
            'timestamp': time.time(),
            'performance': performance
        })
        
        # 保持历史记录长度
        if len(self.strategy_performance[strategy]) > 100:
            self.strategy_performance[strategy] = self.strategy_performance[strategy][-100:]
    
    def get_strategy_recommendations(self) -> Dict[LearningStrategy, float]:
        """获取策略推荐"""
        recommendations = {}
        
        for strategy in self.current_strategies:
            if strategy in self.strategy_performance and self.strategy_performance[strategy]:
                # 计算平均性能
                performances = [p['performance'] for p in self.strategy_performance[strategy]]
                avg_performance = np.mean(performances)
                recommendations[strategy] = avg_performance
            else:
                recommendations[strategy] = 0.5  # 默认分数
        
        return recommendations


class ConsciousnessLearner:
    """F5意识学习器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 初始化各个组件
        self.hierarchy_analyzer = ConsciousnessHierarchyAnalyzer()
        self.state_monitor = ConsciousnessStateMonitor()
        self.associative_engine = ConsciousnessAssociativeEngine()
        self.adaptation_engine = ConsciousnessAdaptationEngine()
        self.evaluation_framework = ConsciousnessEvaluationFramework()
        self.knowledge_base = ConsciousnessKnowledgeBase()
        self.learning_optimizer = ConsciousnessLearningOptimizer()
        
        # 状态变量
        self.is_learning = False
        self.current_patterns = []
        self.current_experiences = []
        self.learning_session_id = None
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def start_learning_session(self, session_id: str):
        """开始学习会话"""
        self.learning_session_id = session_id
        self.is_learning = True
        
        # 启动状态监控
        self.state_monitor.start_monitoring()
        
        # 加载知识库
        self.knowledge_base.load()
        
        self.logger.info(f"开始学习会话: {session_id}")
    
    def end_learning_session(self):
        """结束学习会话"""
        self.is_learning = False
        
        # 停止状态监控
        self.state_monitor.stop_monitoring()
        
        # 清理旧数据
        self.knowledge_base.cleanup_old_data()
        
        self.logger.info(f"结束学习会话: {self.learning_session_id}")
        self.learning_session_id = None
    
    def learn_from_input(self, input_data: Dict[str, Any], 
                        feedback: Optional[float] = None) -> Dict[str, Any]:
        """从输入学习"""
        if not self.is_learning:
            raise RuntimeError("学习会话未启动")
        
        # 1. 创建意识模式
        pattern = self._create_pattern_from_input(input_data)
        
        # 2. 层次分析
        self.hierarchy_analyzer.learn_level_pattern(pattern)
        
        # 3. 存储到知识库
        self.knowledge_base.store_pattern(pattern)
        
        # 4. 创建学习经验
        experience = self._create_learning_experience(input_data, feedback)
        self.knowledge_base.store_experience(experience)
        
        # 5. 关联分析
        self._perform_associative_analysis(pattern)
        
        # 6. 适应性调整
        adapted_patterns = self._perform_adaptation(pattern, feedback)
        
        # 7. 效果评估
        evaluation_results = self._evaluate_learning_effectiveness()
        
        # 8. 策略优化
        optimized_strategy = self._optimize_learning_strategy(input_data, evaluation_results)
        
        result = {
            'pattern': pattern,
            'adapted_patterns': adapted_patterns,
            'evaluation_results': evaluation_results,
            'optimized_strategy': optimized_strategy,
            'associations': self.associative_engine.find_associations(pattern.pattern_id)
        }
        
        return result
    
    def _create_pattern_from_input(self, input_data: Dict[str, Any]) -> ConsciousnessPattern:
        """从输入创建意识模式"""
        # 基于输入数据创建模式
        pattern_id = f"pattern_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # 分析输入特征
        activation_strength = self._calculate_activation_strength(input_data)
        duration = self._estimate_duration(input_data)
        state = self._determine_state(input_data)
        
        return ConsciousnessPattern(
            pattern_id=pattern_id,
            level=ConsciousnessLevel.AWARE,  # 默认层次
            state=state,
            activation_strength=activation_strength,
            duration=duration,
            context=input_data,
            associations=[],
            timestamp=time.time(),
            confidence=0.8  # 默认置信度
        )
    
    def _calculate_activation_strength(self, input_data: Dict[str, Any]) -> float:
        """计算激活强度"""
        # 基于输入复杂度、重要性等计算激活强度
        complexity = len(str(input_data)) / 1000.0  # 简化的复杂度计算
        importance = input_data.get('importance', 0.5)
        
        activation = (complexity * 0.3 + importance * 0.7)
        return min(1.0, activation)
    
    def _estimate_duration(self, input_data: Dict[str, Any]) -> float:
        """估计持续时间"""
        # 基于输入复杂度估计处理持续时间
        complexity = len(str(input_data)) / 100.0
        return max(1.0, complexity)
    
    def _determine_state(self, input_data: Dict[str, Any]) -> ConsciousnessState:
        """确定意识状态"""
        # 基于输入特征确定状态
        urgency = input_data.get('urgency', 0.5)
        creativity_required = input_data.get('creativity_required', False)
        
        if urgency > 0.8:
            return ConsciousnessState.FOCUSED
        elif creativity_required:
            return ConsciousnessState.CREATIVE
        elif urgency < 0.3:
            return ConsciousnessState.PASSIVE
        else:
            return ConsciousnessState.ACTIVE
    
    def _create_learning_experience(self, input_data: Dict[str, Any], 
                                  feedback: Optional[float]) -> LearningExperience:
        """创建学习经验"""
        experience_id = f"exp_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # 模拟输出数据
        output_data = {
            'processed': True,
            'confidence': random.uniform(0.6, 1.0),
            'processing_time': random.uniform(0.1, 2.0)
        }
        
        # 使用提供的反馈或生成模拟反馈
        if feedback is None:
            feedback = random.uniform(-0.5, 1.0)
        
        # 计算成功率
        success_rate = max(0.0, min(1.0, (feedback + 1) / 2))
        
        return LearningExperience(
            experience_id=experience_id,
            input_data=input_data,
            output_data=output_data,
            feedback=feedback,
            context={'session_id': self.learning_session_id},
            timestamp=time.time(),
            success_rate=success_rate,
            learning_impact=success_rate * 0.8
        )
    
    def _perform_associative_analysis(self, pattern: ConsciousnessPattern):
        """执行关联分析"""
        # 与现有模式建立关联
        for existing_pattern in self.current_patterns:
            similarity = self.associative_engine._calculate_pattern_similarity(pattern, existing_pattern)
            if similarity > 0.5:
                self.associative_engine.create_association(
                    pattern.pattern_id, 
                    existing_pattern.pattern_id, 
                    similarity
                )
        
        # 更新当前模式列表
        self.current_patterns.append(pattern)
        
        # 执行聚类
        self.associative_engine.cluster_patterns(self.current_patterns)
    
    def _perform_adaptation(self, pattern: ConsciousnessPattern, 
                          feedback: Optional[float]) -> List[ConsciousnessPattern]:
        """执行适应性调整"""
        if feedback is None:
            return [pattern]
        
        # 构建环境反馈
        environment_feedback = {
            'activation_feedback': feedback,
            'performance_feedback': feedback,
            'adaptation_pressure': abs(feedback)
        }
        
        # 执行适应
        adapted_patterns = self.adaptation_engine.adapt_to_environment(
            [pattern], environment_feedback
        )
        
        return adapted_patterns
    
    def _evaluate_learning_effectiveness(self) -> Dict[str, float]:
        """评估学习效果"""
        return self.evaluation_framework.evaluate_consciousness(
            self.current_patterns, self.current_experiences
        )
    
    def _optimize_learning_strategy(self, input_data: Dict[str, Any], 
                                  evaluation_results: Dict[str, float]) -> LearningStrategy:
        """优化学习策略"""
        # 构建上下文
        context = {
            'input_complexity': len(str(input_data)),
            'input_type': input_data.get('type', 'unknown'),
            'session_duration': time.time() - (self.learning_session_start_time or time.time()),
            'cpu_available': 0.8,
            'memory_available': 0.7,
            'time_available': 0.6,
            'urgency': input_data.get('urgency', 0.5),
            'missing_info': 0.2,
            'conflicting_info': 0.1
        }
        
        # 获取历史性能
        historical_performance = {}
        for strategy in LearningStrategy:
            if strategy in self.learning_optimizer.strategy_performance:
                performances = [p['performance'] for p in self.learning_optimizer.strategy_performance[strategy]]
                historical_performance[strategy.value] = np.mean(performances)
        
        # 优化策略
        optimized_strategy = self.learning_optimizer.optimize_learning_strategy(
            context, historical_performance
        )
        
        # 更新性能历史
        overall_performance = np.mean(list(evaluation_results.values())) if evaluation_results else 0.5
        self.learning_optimizer.update_performance_history(optimized_strategy, overall_performance)
        
        return optimized_strategy
    
    def get_learning_report(self) -> Dict[str, Any]:
        """获取学习报告"""
        if not self.learning_session_id:
            return {'error': '没有活动的学习会话'}
        
        # 收集各种指标
        state_trends = self.state_monitor.analyze_state_trends()
        anomalies = self.state_monitor.detect_anomalies()
        evaluation_results = self._evaluate_learning_effectiveness()
        strategy_recommendations = self.learning_optimizer.get_strategy_recommendations()
        
        # 知识库统计
        kb_stats = {
            'pattern_count': len(self.knowledge_base.patterns),
            'experience_count': len(self.knowledge_base.experiences),
            'rule_count': len(self.knowledge_base.rules),
            'concept_count': len(self.knowledge_base.concepts)
        }
        
        # 层次分析统计
        level_distribution = defaultdict(int)
        for pattern in self.current_patterns:
            level_distribution[pattern.level] += 1
        
        report = {
            'session_id': self.learning_session_id,
            'session_duration': time.time() - getattr(self, 'learning_session_start_time', time.time()),
            'current_patterns': len(self.current_patterns),
            'current_experiences': len(self.current_experiences),
            'state_trends': state_trends,
            'anomalies': anomalies,
            'evaluation_results': evaluation_results,
            'strategy_recommendations': strategy_recommendations,
            'knowledge_base_stats': kb_stats,
            'level_distribution': dict(level_distribution),
            'adaptation_history': self.adaptation_engine.adaptation_history,
            'optimization_history': self.learning_optimizer.optimization_history[-10:]  # 最近10次
        }
        
        return report
    
    def export_knowledge(self, filepath: str):
        """导出知识库"""
        export_data = {
            'patterns': {k: asdict(v) for k, v in self.knowledge_base.patterns.items()},
            'experiences': {k: asdict(v) for k, v in self.knowledge_base.experiences.items()},
            'rules': self.knowledge_base.rules,
            'concepts': self.knowledge_base.concepts,
            'relationships': {k: list(v) for k, v in self.knowledge_base.relationships.items()},
            'export_timestamp': time.time(),
            'session_info': {
                'session_id': self.learning_session_id,
                'export_time': datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"知识库已导出到: {filepath}")
    
    def import_knowledge(self, filepath: str):
        """导入知识库"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 重建对象
            for pattern_id, pattern_data in import_data.get('patterns', {}).items():
                pattern = ConsciousnessPattern(**pattern_data)
                self.knowledge_base.store_pattern(pattern)
            
            for exp_id, exp_data in import_data.get('experiences', {}).items():
                experience = LearningExperience(**exp_data)
                self.knowledge_base.store_experience(experience)
            
            # 导入规则和概念
            for rule_id, rule_data in import_data.get('rules', {}).items():
                self.knowledge_base.add_rule(rule_id, rule_data)
            
            for concept_id, concept_data in import_data.get('concepts', {}).items():
                self.knowledge_base.add_concept(concept_id, concept_data)
            
            self.logger.info(f"知识库已从 {filepath} 导入")
            
        except Exception as e:
            self.logger.error(f"导入知识库失败: {e}")
            raise
    
    def reset_learning_state(self):
        """重置学习状态"""
        self.current_patterns.clear()
        self.current_experiences.clear()
        self.hierarchy_analyzer.level_patterns.clear()
        self.hierarchy_analyzer.transition_matrix = np.zeros((6, 6))
        self.associative_engine.association_network.clear()
        self.associative_engine.pattern_clusters.clear()
        self.adaptation_engine.adaptation_history.clear()
        self.evaluation_framework.evaluation_history.clear()
        self.learning_optimizer.optimization_history.clear()
        
        self.logger.info("学习状态已重置")


# 使用示例和测试代码
def demo_consciousness_learner():
    """演示意识学习器功能"""
    
    # 创建学习器实例
    learner = ConsciousnessLearner()
    
    # 开始学习会话
    learner.start_learning_session("demo_session_001")
    learner.learning_session_start_time = time.time()
    
    print("=== F5意识学习器演示 ===\n")
    
    # 模拟学习过程
    test_inputs = [
        {'type': 'text', 'content': '学习人工智能概念', 'importance': 0.8, 'urgency': 0.6},
        {'type': 'image', 'content': '分析图像模式', 'importance': 0.7, 'urgency': 0.4},
        {'type': 'audio', 'content': '处理音频信号', 'importance': 0.6, 'urgency': 0.7},
        {'type': 'data', 'content': '分析数据趋势', 'importance': 0.9, 'urgency': 0.5},
        {'type': 'code', 'content': '理解代码逻辑', 'importance': 0.8, 'urgency': 0.8}
    ]
    
    learning_results = []
    
    for i, input_data in enumerate(test_inputs):
        print(f"学习第 {i+1} 个输入...")
        
        # 模拟反馈
        feedback = random.uniform(-0.3, 1.0)
        
        # 执行学习
        result = learner.learn_from_input(input_data, feedback)
        learning_results.append(result)
        
        print(f"  - 模式ID: {result['pattern'].pattern_id}")
        print(f"  - 意识层次: {result['pattern'].level}")
        print(f"  - 意识状态: {result['pattern'].state}")
        print(f"  - 激活强度: {result['pattern'].activation_strength:.3f}")
        print(f"  - 关联数量: {len(result['associations'])}")
        print(f"  - 优化策略: {result['optimized_strategy']}")
        print()
    
    # 获取学习报告
    print("=== 学习报告 ===")
    report = learner.get_learning_report()
    
    print(f"会话ID: {report['session_id']}")
    print(f"会话持续时间: {report['session_duration']:.2f} 秒")
    print(f"当前模式数量: {report['current_patterns']}")
    print(f"当前经验数量: {report['current_experiences']}")
    
    print("\n层次分布:")
    for level, count in report['level_distribution'].items():
        print(f"  {level}: {count}")
    
    print("\n评估结果:")
    for metric, score in report['evaluation_results'].items():
        print(f"  {metric}: {score:.3f}")
    
    print("\n策略推荐:")
    for strategy, score in report['strategy_recommendations'].items():
        print(f"  {strategy}: {score:.3f}")
    
    print("\n知识库统计:")
    for stat, value in report['knowledge_base_stats'].items():
        print(f"  {stat}: {value}")
    
    # 导出知识
    learner.export_knowledge("consciousness_knowledge_export.json")
    
    # 结束学习会话
    learner.end_learning_session()
    
    print("\n=== 演示完成 ===")
    
    return learner, learning_results


if __name__ == "__main__":
    # 运行演示
    demo_learner, demo_results = demo_consciousness_learner()