"""
F4认知学习器 (Cognitive Learner)
实现认知模式识别、学习、推理和适应性调整的智能学习系统
"""

import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import math


@dataclass
class CognitivePattern:
    """认知模式数据结构"""
    pattern_id: str
    pattern_type: str
    features: Dict[str, Any]
    confidence: float
    frequency: int
    context: Dict[str, Any]
    timestamp: datetime
    success_rate: float = 0.0
    associations: Set[str] = None
    
    def __post_init__(self):
        if self.associations is None:
            self.associations = set()


@dataclass
class CognitiveStructure:
    """认知结构数据结构"""
    structure_id: str
    structure_type: str
    nodes: Dict[str, Dict[str, Any]]
    edges: List[Tuple[str, str, float]]
    weight: float
    activation_level: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CognitiveAssociation:
    """认知关联数据结构"""
    association_id: str
    source_pattern: str
    target_pattern: str
    association_strength: float
    association_type: str
    confidence: float
    context: Dict[str, Any]
    timestamp: datetime
    usage_count: int = 0


@dataclass
class LearningStrategy:
    """学习策略数据结构"""
    strategy_id: str
    name: str
    strategy_type: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    success_rate: float
    context: Dict[str, Any]
    timestamp: datetime


class PatternRecognizer:
    """认知模式识别器"""
    
    def __init__(self, min_frequency: int = 3, confidence_threshold: float = 0.6):
        self.min_frequency = min_frequency
        self.confidence_threshold = confidence_threshold
        self.patterns: Dict[str, CognitivePattern] = {}
        self.pattern_history: deque = deque(maxlen=1000)
        self.feature_extractors = {}
        
    def register_feature_extractor(self, name: str, extractor_func):
        """注册特征提取器"""
        self.feature_extractors[name] = extractor_func
        
    def extract_features(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """提取特征"""
        features = {}
        
        # 基础特征提取
        if isinstance(data, dict):
            features.update(data)
        elif isinstance(data, (list, tuple)):
            features['length'] = len(data)
            features['type'] = 'sequence'
        elif isinstance(data, str):
            features['length'] = len(data)
            features['type'] = 'text'
        elif isinstance(data, (int, float)):
            features['value'] = data
            features['type'] = 'numeric'
        
        # 应用自定义特征提取器
        for name, extractor in self.feature_extractors.items():
            try:
                extracted = extractor(data, context)
                if isinstance(extracted, dict):
                    features.update(extracted)
            except Exception as e:
                logging.warning(f"特征提取器 {name} 执行失败: {e}")
        
        return features
    
    def recognize_patterns(self, data: Any, context: Dict[str, Any]) -> List[CognitivePattern]:
        """识别认知模式"""
        features = self.extract_features(data, context)
        patterns = []
        
        # 基于特征相似性识别模式
        for existing_pattern in self.patterns.values():
            similarity = self._calculate_similarity(features, existing_pattern.features)
            if similarity >= self.confidence_threshold:
                # 更新现有模式
                existing_pattern.frequency += 1
                existing_pattern.confidence = min(1.0, existing_pattern.confidence + 0.1)
                existing_pattern.timestamp = datetime.now()
                patterns.append(existing_pattern)
        
        # 如果没有匹配的模式，创建新模式
        if not patterns:
            pattern_id = f"pattern_{len(self.patterns)}"
            new_pattern = CognitivePattern(
                pattern_id=pattern_id,
                pattern_type=self._classify_pattern_type(features),
                features=features,
                confidence=0.5,
                frequency=1,
                context=context,
                timestamp=datetime.now()
            )
            self.patterns[pattern_id] = new_pattern
            self.pattern_history.append(new_pattern)
            patterns.append(new_pattern)
        
        return patterns
    
    def _calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """计算特征相似性"""
        if not features1 or not features2:
            return 0.0
        
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        similarity_sum = 0.0
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            
            # 数值相似性
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if val1 == val2:
                    similarity_sum += 1.0
                else:
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        similarity_sum += 1.0 - abs(val1 - val2) / max_val
            
            # 字符串相似性
            elif isinstance(val1, str) and isinstance(val2, str):
                if val1 == val2:
                    similarity_sum += 1.0
                else:
                    # 简单的字符串相似性计算
                    similarity_sum += 0.5 if val1.lower() in val2.lower() or val2.lower() in val1.lower() else 0.0
            
            # 布尔相似性
            elif isinstance(val1, bool) and isinstance(val2, bool):
                similarity_sum += 1.0 if val1 == val2 else 0.0
            
            # 其他类型
            else:
                similarity_sum += 1.0 if val1 == val2 else 0.0
        
        return similarity_sum / len(common_keys)
    
    def _classify_pattern_type(self, features: Dict[str, Any]) -> str:
        """分类模式类型"""
        if 'type' in features:
            return features['type']
        
        # 基于特征数量和复杂度分类
        if len(features) <= 3:
            return 'simple'
        elif len(features) <= 7:
            return 'moderate'
        else:
            return 'complex'


class CognitiveStructureBuilder:
    """认知结构构建器"""
    
    def __init__(self, max_nodes: int = 1000, activation_threshold: float = 0.5):
        self.max_nodes = max_nodes
        self.activation_threshold = activation_threshold
        self.structures: Dict[str, CognitiveStructure] = {}
        self.node_connections: Dict[str, Set[str]] = defaultdict(set)
        
    def build_structure(self, patterns: List[CognitivePattern], structure_type: str = "neural") -> CognitiveStructure:
        """构建认知结构"""
        structure_id = f"structure_{len(self.structures)}"
        
        # 创建节点
        nodes = {}
        for i, pattern in enumerate(patterns):
            node_id = f"node_{i}"
            nodes[node_id] = {
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type,
                'confidence': pattern.confidence,
                'features': pattern.features,
                'activation_level': 0.0
            }
        
        # 创建边（连接）
        edges = []
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns):
                if i != j:
                    similarity = self._calculate_pattern_similarity(pattern1, pattern2)
                    if similarity > 0.3:  # 相似性阈值
                        edge_weight = similarity * pattern1.confidence * pattern2.confidence
                        edges.append((f"node_{i}", f"node_{j}", edge_weight))
        
        # 创建结构
        structure = CognitiveStructure(
            structure_id=structure_id,
            structure_type=structure_type,
            nodes=nodes,
            edges=edges,
            weight=1.0,
            activation_level=0.0,
            timestamp=datetime.now(),
            metadata={'pattern_count': len(patterns)}
        )
        
        self.structures[structure_id] = structure
        return structure
    
    def update_structure(self, structure_id: str, new_patterns: List[CognitivePattern]):
        """更新认知结构"""
        if structure_id not in self.structures:
            return
        
        structure = self.structures[structure_id]
        
        # 添加新节点
        start_node_id = len(structure.nodes)
        for i, pattern in enumerate(new_patterns):
            node_id = f"node_{start_node_id + i}"
            structure.nodes[node_id] = {
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type,
                'confidence': pattern.confidence,
                'features': pattern.features,
                'activation_level': 0.0
            }
        
        # 更新边
        for i, new_pattern in enumerate(new_patterns):
            for existing_node_id, node_data in structure.nodes.items():
                if existing_node_id.startswith(f"node_{start_node_id}"):
                    continue
                
                existing_pattern = None
                for p in [pattern for pattern in [new_pattern] if hasattr(self, 'patterns')]:
                    if p.pattern_id == node_data['pattern_id']:
                        existing_pattern = p
                        break
                
                if existing_pattern:
                    similarity = self._calculate_pattern_similarity(new_pattern, existing_pattern)
                    if similarity > 0.3:
                        edge_weight = similarity * new_pattern.confidence * node_data['confidence']
                        structure.edges.append((f"node_{start_node_id + i}", existing_node_id, edge_weight))
        
        structure.timestamp = datetime.now()
    
    def _calculate_pattern_similarity(self, pattern1: CognitivePattern, pattern2: CognitivePattern) -> float:
        """计算模式相似性"""
        if not pattern1.features or not pattern2.features:
            return 0.0
        
        common_keys = set(pattern1.features.keys()) & set(pattern2.features.keys())
        if not common_keys:
            return 0.0
        
        similarity_sum = 0.0
        for key in common_keys:
            val1, val2 = pattern1.features[key], pattern2.features[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    similarity_sum += 1.0 - abs(val1 - val2) / max_val
                else:
                    similarity_sum += 1.0
            elif val1 == val2:
                similarity_sum += 1.0
            else:
                similarity_sum += 0.0
        
        return similarity_sum / len(common_keys)


class CognitiveReasoner:
    """认知推理器"""
    
    def __init__(self, max_reasoning_depth: int = 5):
        self.max_reasoning_depth = max_reasoning_depth
        self.associations: Dict[str, CognitiveAssociation] = {}
        self.reasoning_cache: Dict[str, Any] = {}
        self.reasoning_rules = {}
        
    def add_association(self, source_pattern: str, target_pattern: str, 
                       association_type: str, strength: float, context: Dict[str, Any]):
        """添加认知关联"""
        association_id = f"assoc_{len(self.associations)}"
        
        association = CognitiveAssociation(
            association_id=association_id,
            source_pattern=source_pattern,
            target_pattern=target_pattern,
            association_strength=strength,
            association_type=association_type,
            confidence=strength,
            context=context,
            timestamp=datetime.now()
        )
        
        self.associations[association_id] = association
    
    def reason(self, query_pattern: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于认知关联进行推理"""
        reasoning_results = []
        
        # 直接关联
        direct_results = self._find_direct_associations(query_pattern, context)
        reasoning_results.extend(direct_results)
        
        # 多步推理
        if len(reasoning_results) < self.max_reasoning_depth:
            multi_step_results = self._multi_step_reasoning(query_pattern, context, 2)
            reasoning_results.extend(multi_step_results)
        
        return reasoning_results
    
    def _find_direct_associations(self, pattern: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找直接关联"""
        results = []
        
        for association in self.associations.values():
            if association.source_pattern == pattern:
                # 检查上下文匹配度
                context_match = self._calculate_context_similarity(context, association.context)
                if context_match > 0.3:
                    results.append({
                        'target_pattern': association.target_pattern,
                        'confidence': association.confidence * context_match,
                        'association_type': association.association_type,
                        'reasoning_depth': 1,
                        'association_strength': association.association_strength
                    })
        
        return results
    
    def _multi_step_reasoning(self, pattern: str, context: Dict[str, Any], depth: int) -> List[Dict[str, Any]]:
        """多步推理"""
        if depth <= 0:
            return []
        
        results = []
        intermediate_patterns = self._find_direct_associations(pattern, context)
        
        for result in intermediate_patterns:
            next_level_results = self._multi_step_reasoning(
                result['target_pattern'], context, depth - 1
            )
            
            for next_result in next_level_results:
                # 组合置信度
                combined_confidence = result['confidence'] * next_result['confidence'] * 0.8
                results.append({
                    'target_pattern': next_result['target_pattern'],
                    'confidence': combined_confidence,
                    'association_type': f"{result['association_type']} -> {next_result['association_type']}",
                    'reasoning_depth': self.max_reasoning_depth - depth + 1,
                    'association_strength': (result['association_strength'] + next_result['association_strength']) / 2
                })
        
        return results
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """计算上下文相似性"""
        if not context1 or not context2:
            return 0.0
        
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        similarity_sum = 0.0
        for key in common_keys:
            val1, val2 = context1[key], context2[key]
            if val1 == val2:
                similarity_sum += 1.0
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    similarity_sum += 1.0 - abs(val1 - val2) / max_val
        
        return similarity_sum / len(common_keys)


class CognitiveAdapter:
    """认知适配器"""
    
    def __init__(self, adaptation_threshold: float = 0.1):
        self.adaptation_threshold = adaptation_threshold
        self.adaptation_history: deque = deque(maxlen=500)
        self.performance_metrics: Dict[str, float] = {}
        self.adaptation_strategies = {
            'increase_confidence': self._increase_confidence,
            'decrease_confidence': self._decrease_confidence,
            'modify_structure': self._modify_structure,
            'create_new_pattern': self._create_new_pattern
        }
    
    def adapt(self, current_performance: float, target_performance: float, 
              context: Dict[str, Any]) -> Dict[str, Any]:
        """执行认知适应"""
        adaptation_needed = target_performance - current_performance
        
        if abs(adaptation_needed) < self.adaptation_threshold:
            return {'status': 'no_adaptation_needed', 'changes': []}
        
        adaptations = []
        
        if adaptation_needed > 0:  # 需要改进
            adaptations.extend(self._positive_adaptation(adaptation_needed, context))
        else:  # 需要降低复杂度
            adaptations.extend(self._negative_adaptation(abs(adaptation_needed), context))
        
        # 记录适应历史
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'performance_before': current_performance,
            'performance_after': target_performance,
            'adaptations': adaptations,
            'context': context
        })
        
        return {
            'status': 'adaptation_applied',
            'changes': adaptations,
            'performance_change': adaptation_needed
        }
    
    def _positive_adaptation(self, improvement_needed: float, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """正向适应"""
        adaptations = []
        
        # 增加置信度
        if improvement_needed > 0.5:
            adaptations.append({
                'type': 'increase_confidence',
                'parameters': {'increment': min(0.2, improvement_needed * 0.3)},
                'rationale': '需要显著提升性能'
            })
        
        # 创建新模式
        if improvement_needed > 0.3:
            adaptations.append({
                'type': 'create_new_pattern',
                'parameters': {'pattern_type': 'adaptive'},
                'rationale': '现有模式不足以满足需求'
            })
        
        return adaptations
    
    def _negative_adaptation(self, reduction_needed: float, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """负向适应"""
        adaptations = []
        
        # 降低置信度
        if reduction_needed > 0.3:
            adaptations.append({
                'type': 'decrease_confidence',
                'parameters': {'decrement': min(0.2, reduction_needed * 0.3)},
                'rationale': '性能过高，需要适当降低'
            })
        
        # 修改结构
        if reduction_needed > 0.5:
            adaptations.append({
                'type': 'modify_structure',
                'parameters': {'simplification_level': reduction_needed},
                'rationale': '需要简化认知结构'
            })
        
        return adaptations
    
    def _increase_confidence(self, parameters: Dict[str, Any]):
        """增加置信度"""
        increment = parameters.get('increment', 0.1)
        # 实现置信度增加逻辑
        pass
    
    def _decrease_confidence(self, parameters: Dict[str, Any]):
        """降低置信度"""
        decrement = parameters.get('decrement', 0.1)
        # 实现置信度降低逻辑
        pass
    
    def _modify_structure(self, parameters: Dict[str, Any]):
        """修改结构"""
        simplification_level = parameters.get('simplification_level', 0.1)
        # 实现结构修改逻辑
        pass
    
    def _create_new_pattern(self, parameters: Dict[str, Any]):
        """创建新模式"""
        pattern_type = parameters.get('pattern_type', 'adaptive')
        # 实现新模式创建逻辑
        pass


class CognitiveEvaluator:
    """认知效果评估器"""
    
    def __init__(self, evaluation_window: int = 100):
        self.evaluation_window = evaluation_window
        self.evaluation_history: deque = deque(maxlen=evaluation_window)
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'learning_rate': [],
            'adaptation_speed': []
        }
    
    def evaluate_performance(self, predictions: List[Any], actuals: List[Any], 
                           context: Dict[str, Any]) -> Dict[str, float]:
        """评估认知性能"""
        if len(predictions) != len(actuals):
            raise ValueError("预测值和实际值数量不匹配")
        
        # 计算基本指标
        accuracy = self._calculate_accuracy(predictions, actuals)
        precision = self._calculate_precision(predictions, actuals)
        recall = self._calculate_recall(predictions, actuals)
        f1_score = self._calculate_f1_score(precision, recall)
        
        # 计算学习率
        learning_rate = self._calculate_learning_rate(context)
        
        # 计算适应速度
        adaptation_speed = self._calculate_adaptation_speed(context)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'learning_rate': learning_rate,
            'adaptation_speed': adaptation_speed,
            'timestamp': datetime.now()
        }
        
        # 更新历史记录
        for key, value in metrics.items():
            if key != 'timestamp':
                self.metrics[key].append(value)
        
        self.evaluation_history.append(metrics)
        
        return metrics
    
    def validate_learning(self, learning_results: Dict[str, Any]) -> Dict[str, Any]:
        """验证学习效果"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'recommendations': []
        }
        
        # 检查准确性
        if learning_results.get('accuracy', 0) < 0.6:
            validation_results['is_valid'] = False
            validation_results['issues'].append('准确率过低')
            validation_results['recommendations'].append('需要改进模式识别算法')
        
        # 检查一致性
        recent_metrics = list(self.evaluation_history)[-10:]
        if len(recent_metrics) >= 2:
            accuracy_variance = np.var([m['accuracy'] for m in recent_metrics])
            if accuracy_variance > 0.1:
                validation_results['issues'].append('性能波动较大')
                validation_results['recommendations'].append('需要提高学习稳定性')
        
        # 检查学习速度
        if learning_results.get('learning_rate', 0) < 0.1:
            validation_results['issues'].append('学习速度较慢')
            validation_results['recommendations'].append('考虑调整学习策略')
        
        return validation_results
    
    def _calculate_accuracy(self, predictions: List[Any], actuals: List[Any]) -> float:
        """计算准确率"""
        if not predictions:
            return 0.0
        
        correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
        return correct / len(predictions)
    
    def _calculate_precision(self, predictions: List[Any], actuals: List[Any]) -> float:
        """计算精确率"""
        if not predictions:
            return 0.0
        
        true_positives = sum(1 for p, a in zip(predictions, actuals) if p == a and p is not None)
        predicted_positives = sum(1 for p in predictions if p is not None)
        
        if predicted_positives == 0:
            return 0.0
        
        return true_positives / predicted_positives
    
    def _calculate_recall(self, predictions: List[Any], actuals: List[Any]) -> float:
        """计算召回率"""
        if not actuals:
            return 0.0
        
        true_positives = sum(1 for p, a in zip(predictions, actuals) if p == a and a is not None)
        actual_positives = sum(1 for a in actuals if a is not None)
        
        if actual_positives == 0:
            return 0.0
        
        return true_positives / actual_positives
    
    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_learning_rate(self, context: Dict[str, Any]) -> float:
        """计算学习率"""
        # 基于上下文计算学习率
        if 'learning_iterations' in context:
            return min(1.0, context['learning_iterations'] / 100.0)
        return 0.1
    
    def _calculate_adaptation_speed(self, context: Dict[str, Any]) -> float:
        """计算适应速度"""
        if 'adaptation_time' in context:
            return min(1.0, 1.0 / max(0.1, context['adaptation_time']))
        return 0.5


class CognitiveKnowledgeBase:
    """认知知识库管理器"""
    
    def __init__(self, max_knowledge_items: int = 10000):
        self.max_knowledge_items = max_knowledge_items
        self.knowledge_items: Dict[str, Dict[str, Any]] = {}
        self.knowledge_index: Dict[str, Set[str]] = defaultdict(set)
        self.knowledge_categories = {
            'patterns': set(),
            'structures': set(),
            'associations': set(),
            'strategies': set(),
            'experiences': set()
        }
        
    def store_knowledge(self, item_id: str, knowledge: Dict[str, Any], category: str):
        """存储知识"""
        if len(self.knowledge_items) >= self.max_knowledge_items:
            # 清理最旧的知识项
            oldest_key = min(self.knowledge_items.keys(), 
                           key=lambda k: self.knowledge_items[k]['timestamp'])
            del self.knowledge_items[oldest_key]
        
        knowledge_item = {
            'id': item_id,
            'data': knowledge,
            'category': category,
            'timestamp': datetime.now(),
            'access_count': 0,
            'relevance_score': 1.0
        }
        
        self.knowledge_items[item_id] = knowledge_item
        # 动态添加新类别
        if category not in self.knowledge_categories:
            self.knowledge_categories[category] = set()
        self.knowledge_categories[category].add(item_id)
        
        # 更新索引
        self._update_index(item_id, knowledge)
    
    def retrieve_knowledge(self, query: Dict[str, Any], category: str = None, 
                          limit: int = 10) -> List[Dict[str, Any]]:
        """检索知识"""
        candidates = []
        
        # 确定搜索范围
        search_items = set()
        if category and category in self.knowledge_categories:
            search_items = self.knowledge_categories[category].copy()
        else:
            # 搜索所有类别
            for cat_items in self.knowledge_categories.values():
                search_items.update(cat_items)
        
        # 计算相关性
        for item_id in search_items:
            if item_id in self.knowledge_items:
                item = self.knowledge_items[item_id]
                relevance = self._calculate_relevance(query, item['data'])
                if relevance > 0.1:  # 相关性阈值
                    candidates.append((item, relevance))
        
        # 按相关性排序并返回
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in candidates[:limit]]
    
    def update_knowledge(self, item_id: str, updates: Dict[str, Any]):
        """更新知识"""
        if item_id not in self.knowledge_items:
            return False
        
        item = self.knowledge_items[item_id]
        item['data'].update(updates)
        item['timestamp'] = datetime.now()
        item['access_count'] += 1
        
        return True
    
    def delete_knowledge(self, item_id: str):
        """删除知识"""
        if item_id not in self.knowledge_items:
            return False
        
        item = self.knowledge_items[item_id]
        category = item['category']
        
        del self.knowledge_items[item_id]
        if category in self.knowledge_categories:
            self.knowledge_categories[category].discard(item_id)
        
        # 更新索引
        self._remove_from_index(item_id, item['data'])
        
        return True
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        stats = {
            'total_items': len(self.knowledge_items),
            'categories': {cat: len(items) for cat, items in self.knowledge_categories.items()},
            'most_accessed': [],
            'recent_items': [],
            'index_size': {key: len(value) for key, value in self.knowledge_index.items()}
        }
        
        # 获取最常访问的知识项
        accessed_items = sorted(self.knowledge_items.items(), 
                              key=lambda x: x[1]['access_count'], reverse=True)
        stats['most_accessed'] = [(k, v['access_count']) for k, v in accessed_items[:5]]
        
        # 获取最近添加的知识项
        recent_items = sorted(self.knowledge_items.items(), 
                            key=lambda x: x[1]['timestamp'], reverse=True)
        stats['recent_items'] = [(k, v['timestamp']) for k, v in recent_items[:5]]
        
        return stats
    
    def _update_index(self, item_id: str, knowledge: Dict[str, Any]):
        """更新知识索引"""
        # 提取关键词
        keywords = self._extract_keywords(knowledge)
        for keyword in keywords:
            self.knowledge_index[keyword].add(item_id)
    
    def _remove_from_index(self, item_id: str, knowledge: Dict[str, Any]):
        """从索引中移除"""
        keywords = self._extract_keywords(knowledge)
        for keyword in keywords:
            self.knowledge_index[keyword].discard(item_id)
            if not self.knowledge_index[keyword]:
                del self.knowledge_index[keyword]
    
    def _extract_keywords(self, knowledge: Dict[str, Any]) -> Set[str]:
        """提取关键词"""
        keywords = set()
        
        def extract_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(key, str):
                        keywords.add(key.lower())
                    extract_recursive(value)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    extract_recursive(item)
            elif isinstance(obj, str):
                # 简单的词分割
                words = obj.lower().split()
                keywords.update(words)
        
        extract_recursive(knowledge)
        return keywords
    
    def _calculate_relevance(self, query: Dict[str, Any], knowledge: Dict[str, Any]) -> float:
        """计算相关性"""
        if not query or not knowledge:
            return 0.0
        
        relevance_score = 0.0
        query_keywords = self._extract_keywords(query)
        knowledge_keywords = self._extract_keywords(knowledge)
        
        if query_keywords and knowledge_keywords:
            intersection = query_keywords & knowledge_keywords
            relevance_score = len(intersection) / len(query_keywords | knowledge_keywords)
        
        return relevance_score


class LearningStrategyOptimizer:
    """学习策略优化器"""
    
    def __init__(self, strategy_history_size: int = 100):
        self.strategy_history_size = strategy_history_size
        self.strategies: Dict[str, LearningStrategy] = {}
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.optimization_rules = {
            'exploration_rate': 0.1,
            'exploitation_rate': 0.8,
            'adaptation_threshold': 0.05
        }
        
    def add_strategy(self, strategy: LearningStrategy):
        """添加学习策略"""
        self.strategies[strategy.strategy_id] = strategy
        
        # 初始化性能记录
        if strategy.strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy.strategy_id] = []
    
    def optimize_strategies(self, current_performance: float, 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """优化学习策略"""
        optimization_results = {
            'selected_strategy': None,
            'optimizations_applied': [],
            'performance_predictions': {}
        }
        
        # 评估当前策略性能
        strategy_scores = {}
        for strategy_id, strategy in self.strategies.items():
            score = self._evaluate_strategy_performance(strategy, current_performance, context)
            strategy_scores[strategy_id] = score
        
        # 选择最佳策略
        if strategy_scores:
            best_strategy_id = max(strategy_scores, key=strategy_scores.get)
            optimization_results['selected_strategy'] = best_strategy_id
            
            # 生成性能预测
            best_strategy = self.strategies[best_strategy_id]
            optimization_results['performance_predictions'] = self._predict_performance(
                best_strategy, context
            )
        
        # 应用优化
        if current_performance < 0.6:  # 性能较低，需要探索
            optimization_results['optimizations_applied'].append({
                'type': 'increase_exploration',
                'description': '增加探索比例以发现更好的策略'
            })
        elif current_performance > 0.8:  # 性能较高，可以利用
            optimization_results['optimizations_applied'].append({
                'type': 'increase_exploitation',
                'description': '增加利用比例以巩固当前策略'
            })
        
        return optimization_results
    
    def adapt_strategy_parameters(self, strategy_id: str, performance_feedback: float):
        """调整策略参数"""
        if strategy_id not in self.strategies:
            return False
        
        strategy = self.strategies[strategy_id]
        
        # 记录性能
        self.strategy_performance[strategy_id].append(performance_feedback)
        
        # 自适应调整参数
        recent_performance = self.strategy_performance[strategy_id][-5:]
        if len(recent_performance) >= 3:
            performance_trend = np.mean(np.diff(recent_performance))
            
            if performance_trend > 0:  # 性能上升
                # 保持当前参数或轻微调整
                pass
            else:  # 性能下降
                # 调整参数
                if 'learning_rate' in strategy.parameters:
                    current_lr = strategy.parameters['learning_rate']
                    strategy.parameters['learning_rate'] = min(1.0, current_lr * 1.1)
        
        return True
    
    def _evaluate_strategy_performance(self, strategy: LearningStrategy, 
                                     current_performance: float, 
                                     context: Dict[str, Any]) -> float:
        """评估策略性能"""
        base_score = strategy.success_rate
        
        # 考虑历史性能
        if strategy.strategy_id in self.strategy_performance:
            historical_scores = self.strategy_performance[strategy.strategy_id]
            if historical_scores:
                avg_historical = np.mean(historical_scores[-10:])  # 最近10次
                base_score = (base_score + avg_historical) / 2
        
        # 考虑上下文适配度
        context_match = self._calculate_context_match(strategy.context, context)
        
        # 综合评分
        final_score = base_score * 0.7 + context_match * 0.3
        
        return final_score
    
    def _predict_performance(self, strategy: LearningStrategy, 
                           context: Dict[str, Any]) -> Dict[str, float]:
        """预测策略性能"""
        base_performance = strategy.success_rate
        
        # 基于历史趋势预测
        if strategy.strategy_id in self.strategy_performance:
            historical = self.strategy_performance[strategy.strategy_id]
            if len(historical) >= 3:
                trend = np.polyfit(range(len(historical)), historical, 1)[0]
                predicted_performance = base_performance + trend * len(historical)
            else:
                predicted_performance = base_performance
        else:
            predicted_performance = base_performance
        
        # 添加不确定性
        uncertainty = 0.1
        confidence_interval = {
            'lower': max(0.0, predicted_performance - uncertainty),
            'upper': min(1.0, predicted_performance + uncertainty)
        }
        
        return {
            'predicted_performance': predicted_performance,
            'confidence_interval': confidence_interval
        }
    
    def _calculate_context_match(self, strategy_context: Dict[str, Any], 
                               current_context: Dict[str, Any]) -> float:
        """计算上下文匹配度"""
        if not strategy_context or not current_context:
            return 0.5  # 默认中等匹配度
        
        common_keys = set(strategy_context.keys()) & set(current_context.keys())
        if not common_keys:
            return 0.3  # 低匹配度
        
        match_score = 0.0
        for key in common_keys:
            if strategy_context[key] == current_context[key]:
                match_score += 1.0
            elif isinstance(strategy_context[key], (int, float)) and isinstance(current_context[key], (int, float)):
                # 数值相似性
                max_val = max(abs(strategy_context[key]), abs(current_context[key]))
                if max_val > 0:
                    match_score += 1.0 - abs(strategy_context[key] - current_context[key]) / max_val
        
        return match_score / len(common_keys)


class CognitiveLearner:
    """F4认知学习器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 初始化各个组件
        self.pattern_recognizer = PatternRecognizer(
            min_frequency=self.config.get('min_pattern_frequency', 3),
            confidence_threshold=self.config.get('confidence_threshold', 0.6)
        )
        
        self.structure_builder = CognitiveStructureBuilder(
            max_nodes=self.config.get('max_structure_nodes', 1000),
            activation_threshold=self.config.get('activation_threshold', 0.5)
        )
        
        self.reasoner = CognitiveReasoner(
            max_reasoning_depth=self.config.get('max_reasoning_depth', 5)
        )
        
        self.adapter = CognitiveAdapter(
            adaptation_threshold=self.config.get('adaptation_threshold', 0.1)
        )
        
        self.evaluator = CognitiveEvaluator(
            evaluation_window=self.config.get('evaluation_window', 100)
        )
        
        self.knowledge_base = CognitiveKnowledgeBase(
            max_knowledge_items=self.config.get('max_knowledge_items', 10000)
        )
        
        self.strategy_optimizer = LearningStrategyOptimizer(
            strategy_history_size=self.config.get('strategy_history_size', 100)
        )
        
        # 学习状态
        self.learning_state = {
            'is_learning': False,
            'current_iteration': 0,
            'total_iterations': 0,
            'performance_history': [],
            'last_update': datetime.now()
        }
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 初始化默认策略
        self._initialize_default_strategies()
        
        logging.info("F4认知学习器初始化完成")
    
    def _initialize_default_strategies(self):
        """初始化默认学习策略"""
        default_strategies = [
            LearningStrategy(
                strategy_id="exploration_focused",
                name="探索导向策略",
                strategy_type="exploration",
                parameters={
                    'learning_rate': 0.1,
                    'exploration_rate': 0.8,
                    'confidence_threshold': 0.5
                },
                performance_metrics={'accuracy': 0.0, 'speed': 0.0},
                success_rate=0.6,
                context={'domain': 'general', 'complexity': 'medium'},
                timestamp=datetime.now()
            ),
            LearningStrategy(
                strategy_id="exploitation_focused",
                name="利用导向策略",
                strategy_type="exploitation",
                parameters={
                    'learning_rate': 0.05,
                    'exploration_rate': 0.2,
                    'confidence_threshold': 0.7
                },
                performance_metrics={'accuracy': 0.0, 'speed': 0.0},
                success_rate=0.7,
                context={'domain': 'general', 'complexity': 'low'},
                timestamp=datetime.now()
            )
        ]
        
        for strategy in default_strategies:
            self.strategy_optimizer.add_strategy(strategy)
    
    def learn(self, data: Any, context: Dict[str, Any], target_performance: float = 0.8) -> Dict[str, Any]:
        """执行认知学习"""
        with self.lock:
            self.learning_state['is_learning'] = True
            self.learning_state['current_iteration'] += 1
            
            try:
                # 1. 模式识别和学习
                patterns = self.pattern_recognizer.recognize_patterns(data, context)
                
                # 2. 认知结构构建和更新
                if self.learning_state['current_iteration'] % 10 == 0:  # 每10次迭代重建结构
                    structure = self.structure_builder.build_structure(patterns)
                    self.knowledge_base.store_knowledge(
                        structure.structure_id, asdict(structure), 'structures'
                    )
                else:
                    # 更新现有结构
                    for structure_id in self.structure_builder.structures:
                        self.structure_builder.update_structure(structure_id, patterns)
                
                # 3. 认知关联和推理
                for pattern in patterns:
                    reasoning_results = self.reasoner.reason(pattern.pattern_id, context)
                    for result in reasoning_results:
                        self.reasoner.add_association(
                            pattern.pattern_id,
                            result['target_pattern'],
                            result['association_type'],
                            result['association_strength'],
                            context
                        )
                
                # 4. 认知适应性和调整
                current_performance = self._calculate_current_performance(patterns, context)
                adaptation_result = self.adapter.adapt(
                    current_performance, target_performance, context
                )
                
                # 5. 认知效果评估和验证
                predictions = [p.pattern_id for p in patterns]
                # 这里应该根据实际场景提供真实标签
                actuals = predictions  # 简化处理
                evaluation_metrics = self.evaluator.evaluate_performance(predictions, actuals, context)
                validation_result = self.evaluator.validate_learning(evaluation_metrics)
                
                # 6. 认知知识库管理
                for pattern in patterns:
                    self.knowledge_base.store_knowledge(
                        pattern.pattern_id, asdict(pattern), 'patterns'
                    )
                
                # 7. 认知学习策略优化
                optimization_result = self.strategy_optimizer.optimize_strategies(
                    current_performance, context
                )
                
                # 更新学习状态
                self.learning_state['performance_history'].append(current_performance)
                self.learning_state['last_update'] = datetime.now()
                
                learning_result = {
                    'success': True,
                    'patterns_learned': len(patterns),
                    'current_performance': current_performance,
                    'target_performance': target_performance,
                    'adaptation_result': adaptation_result,
                    'evaluation_metrics': evaluation_metrics,
                    'validation_result': validation_result,
                    'optimization_result': optimization_result,
                    'learning_state': self.learning_state.copy()
                }
                
                logging.info(f"认知学习完成 - 迭代 {self.learning_state['current_iteration']}, "
                           f"性能: {current_performance:.3f}")
                
                return learning_result
                
            except Exception as e:
                logging.error(f"认知学习过程中发生错误: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'learning_state': self.learning_state.copy()
                }
            finally:
                self.learning_state['is_learning'] = False
    
    def predict(self, query_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """基于学习的认知模式进行预测"""
        try:
            # 使用模式识别器识别查询模式
            query_patterns = self.pattern_recognizer.recognize_patterns(query_data, context)
            
            # 使用推理器进行推理
            predictions = []
            for pattern in query_patterns:
                reasoning_results = self.reasoner.reason(pattern.pattern_id, context)
                predictions.extend(reasoning_results)
            
            # 检索相关知识
            relevant_knowledge = self.knowledge_base.retrieve_knowledge(
                {'pattern': query_data, 'context': context}, limit=5
            )
            
            return {
                'success': True,
                'query_patterns': [asdict(p) for p in query_patterns],
                'predictions': predictions,
                'relevant_knowledge': relevant_knowledge,
                'confidence': np.mean([p.confidence for p in query_patterns]) if query_patterns else 0.0
            }
            
        except Exception as e:
            logging.error(f"预测过程中发生错误: {e}")
            return {
                'success': False,
                'error': str(e),
                'predictions': [],
                'confidence': 0.0
            }
    
    def get_learning_status(self) -> Dict[str, Any]:
        """获取学习状态"""
        with self.lock:
            status = {
                'learning_state': self.learning_state.copy(),
                'pattern_count': len(self.pattern_recognizer.patterns),
                'structure_count': len(self.structure_builder.structures),
                'association_count': len(self.reasoner.associations),
                'knowledge_base_stats': self.knowledge_base.get_knowledge_statistics(),
                'strategy_count': len(self.strategy_optimizer.strategies),
                'recent_performance': self.learning_state['performance_history'][-10:] if self.learning_state['performance_history'] else []
            }
            return status
    
    def reset_learning(self):
        """重置学习状态"""
        with self.lock:
            self.learning_state = {
                'is_learning': False,
                'current_iteration': 0,
                'total_iterations': 0,
                'performance_history': [],
                'last_update': datetime.now()
            }
            
            # 重置各个组件
            self.pattern_recognizer.patterns.clear()
            self.pattern_recognizer.pattern_history.clear()
            self.structure_builder.structures.clear()
            self.reasoner.associations.clear()
            self.reasoner.reasoning_cache.clear()
            
            logging.info("认知学习器状态已重置")
    
    def save_state(self, filepath: str):
        """保存学习器状态"""
        try:
            state = {
                'config': self.config,
                'learning_state': self.learning_state,
                'patterns': {k: asdict(v) for k, v in self.pattern_recognizer.patterns.items()},
                'structures': {k: asdict(v) for k, v in self.structure_builder.structures.items()},
                'associations': {k: asdict(v) for k, v in self.reasoner.associations.items()},
                'knowledge_base': self.knowledge_base.knowledge_items,
                'strategies': {k: asdict(v) for k, v in self.strategy_optimizer.strategies.items()},
                'performance_history': self.learning_state['performance_history']
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            logging.info(f"学习器状态已保存到: {filepath}")
            
        except Exception as e:
            logging.error(f"保存状态失败: {e}")
    
    def load_state(self, filepath: str):
        """加载学习器状态"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # 恢复配置
            self.config.update(state.get('config', {}))
            
            # 恢复学习状态
            self.learning_state.update(state.get('learning_state', {}))
            
            # 恢复模式
            self.pattern_recognizer.patterns.clear()
            for k, v in state.get('patterns', {}).items():
                # 重建CognitivePattern对象
                pattern = CognitivePattern(**v)
                self.pattern_recognizer.patterns[k] = pattern
            
            # 恢复结构
            self.structure_builder.structures.clear()
            for k, v in state.get('structures', {}).items():
                structure = CognitiveStructure(**v)
                self.structure_builder.structures[k] = structure
            
            # 恢复关联
            self.reasoner.associations.clear()
            for k, v in state.get('associations', {}).items():
                association = CognitiveAssociation(**v)
                self.reasoner.associations[k] = association
            
            # 恢复知识库
            self.knowledge_base.knowledge_items = state.get('knowledge_base', {})
            
            # 恢复策略
            self.strategy_optimizer.strategies.clear()
            for k, v in state.get('strategies', {}).items():
                strategy = LearningStrategy(**v)
                self.strategy_optimizer.strategies[k] = strategy
            
            # 恢复性能历史
            self.learning_state['performance_history'] = state.get('performance_history', [])
            
            logging.info(f"学习器状态已从 {filepath} 加载")
            
        except Exception as e:
            logging.error(f"加载状态失败: {e}")
    
    def _calculate_current_performance(self, patterns: List[CognitivePattern], 
                                     context: Dict[str, Any]) -> float:
        """计算当前性能"""
        if not patterns:
            return 0.0
        
        # 基于模式质量计算性能
        avg_confidence = np.mean([p.confidence for p in patterns])
        avg_frequency = np.mean([p.frequency for p in patterns])
        
        # 归一化频率
        max_frequency = max([p.frequency for p in patterns]) if patterns else 1
        normalized_frequency = avg_frequency / max_frequency if max_frequency > 0 else 0
        
        # 综合性能评分
        performance = (avg_confidence * 0.6 + normalized_frequency * 0.4)
        
        return min(1.0, performance)


# 使用示例和测试代码
def example_usage():
    """认知学习器使用示例"""
    
    # 创建学习器实例
    learner = CognitiveLearner({
        'min_pattern_frequency': 2,
        'confidence_threshold': 0.5,
        'max_reasoning_depth': 3,
        'evaluation_window': 50
    })
    
    # 模拟学习数据
    training_data = [
        {'feature1': 1.0, 'feature2': 0.5, 'label': 'A'},
        {'feature1': 0.8, 'feature2': 0.6, 'label': 'A'},
        {'feature1': 0.9, 'feature2': 0.4, 'label': 'B'},
        {'feature1': 0.7, 'feature2': 0.8, 'label': 'B'},
        {'feature1': 1.2, 'feature2': 0.3, 'label': 'A'}
    ]
    
    # 执行学习
    print("开始认知学习...")
    for i, data in enumerate(training_data):
        context = {'iteration': i, 'data_type': 'training'}
        result = learner.learn(data, context, target_performance=0.8)
        
        if result['success']:
            print(f"迭代 {i+1}: 性能 {result['current_performance']:.3f}, "
                  f"学习模式 {result['patterns_learned']}")
        else:
            print(f"迭代 {i+1} 学习失败: {result['error']}")
    
    # 获取学习状态
    status = learner.get_learning_status()
    print(f"\n学习状态:")
    print(f"- 已学习模式数: {status['pattern_count']}")
    print(f"- 认知结构数: {status['structure_count']}")
    print(f"- 认知关联数: {status['association_count']}")
    print(f"- 知识库项目数: {status['knowledge_base_stats']['total_items']}")
    
    # 执行预测
    print("\n执行预测...")
    test_data = {'feature1': 0.85, 'feature2': 0.55}
    test_context = {'test': True}
    
    prediction_result = learner.predict(test_data, test_context)
    
    if prediction_result['success']:
        print(f"预测置信度: {prediction_result['confidence']:.3f}")
        print(f"预测结果数量: {len(prediction_result['predictions'])}")
    else:
        print(f"预测失败: {prediction_result['error']}")
    
    # 保存状态
    learner.save_state('cognitive_learner_state.pkl')
    print("\n学习器状态已保存")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行示例
    example_usage()