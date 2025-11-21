#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1深度反思器 - DeepReflection

实现深度反思分析框架，具备反思内容收集、模式识别、学习、评估、
验证、经验提取、效果跟踪、历史管理和报告生成等功能。


版本: 1.0.0
创建时间: 2025-11-05
"""

import json
import sqlite3
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from pathlib import Path
import re
import hashlib
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReflectionContent:
    """反思内容数据结构"""
    id: str
    timestamp: datetime.datetime
    content_type: str  # 'thought', 'action', 'result', 'emotion'
    content: str
    context: Dict[str, Any]
    tags: List[str]
    importance_score: float
    confidence_score: float
    emotional_valence: float  # -1到1，负面到正面
    emotional_arousal: float  # 0到1，平静到激动
    source: str
    metadata: Dict[str, Any]

@dataclass
class ReflectionPattern:
    """反思模式数据结构"""
    pattern_id: str
    pattern_type: str  # 'temporal', 'thematic', 'behavioral', 'emotional'
    pattern_description: str
    frequency: int
    confidence: float
    supporting_content_ids: List[str]
    pattern_strength: float
    first_observed: datetime.datetime
    last_observed: datetime.datetime
    metadata: Dict[str, Any]

@dataclass
class ReflectionInsight:
    """反思洞察数据结构"""
    insight_id: str
    insight_type: str  # 'pattern', 'correlation', 'trend', 'anomaly'
    title: str
    description: str
    supporting_evidence: List[str]
    confidence_score: float
    impact_score: float
    actionable: bool
    recommendations: List[str]
    created_at: datetime.datetime
    validated: bool
    validation_score: float

@dataclass
class ReflectionExperience:
    """反思经验数据结构"""
    experience_id: str
    experience_type: str  # 'success', 'failure', 'learning', 'adaptation'
    title: str
    description: str
    context: Dict[str, Any]
    lessons_learned: List[str]
    best_practices: List[str]
    pitfalls: List[str]
    applicable_scenarios: List[str]
    confidence_level: float
    created_at: datetime.datetime
    last_applied: Optional[datetime.datetime]
    usage_count: int
    effectiveness_score: float

class ReflectionContentCollector:
    """反思内容收集器"""
    
    def __init__(self):
        self.collection_strategies = {
            'manual': self._collect_manual_content,
            'automatic': self._collect_automatic_content,
            'sensors': self._collect_sensor_content,
            'interaction': self._collect_interaction_content
        }
        
    def collect_content(self, content_type: str, source: str, 
                       content: str, context: Dict[str, Any] = None) -> ReflectionContent:
        """收集反思内容"""
        try:
            collector = self.collection_strategies.get(source, self._collect_manual_content)
            return collector(content_type, content, context or {})
        except Exception as e:
            logger.error(f"内容收集失败: {e}")
            raise
    
    def _collect_manual_content(self, content_type: str, content: str, 
                               context: Dict[str, Any]) -> ReflectionContent:
        """手动收集内容"""
        content_id = self._generate_content_id(content)
        
        # 分析内容特征
        importance_score = self._calculate_importance(content)
        confidence_score = self._calculate_confidence(content, context)
        emotional_features = self._analyze_emotion(content)
        
        return ReflectionContent(
            id=content_id,
            timestamp=datetime.datetime.now(),
            content_type=content_type,
            content=content,
            context=context,
            tags=self._extract_tags(content),
            importance_score=importance_score,
            confidence_score=confidence_score,
            emotional_valence=emotional_features['valence'],
            emotional_arousal=emotional_features['arousal'],
            source='manual',
            metadata=context
        )
    
    def _collect_automatic_content(self, content_type: str, content: str, 
                                  context: Dict[str, Any]) -> ReflectionContent:
        """自动收集内容"""
        # 自动收集通常来自系统日志、行为记录等
        return self._collect_manual_content(content_type, content, context)
    
    def _collect_sensor_content(self, content_type: str, content: str, 
                               context: Dict[str, Any]) -> ReflectionContent:
        """传感器数据收集"""
        return self._collect_manual_content(content_type, content, context)
    
    def _collect_interaction_content(self, content_type: str, content: str, 
                                    context: Dict[str, Any]) -> ReflectionContent:
        """交互数据收集"""
        return self._collect_manual_content(content_type, content, context)
    
    def _generate_content_id(self, content: str) -> str:
        """生成内容ID"""
        return hashlib.md5(f"{content}{datetime.datetime.now()}".encode()).hexdigest()[:16]
    
    def _calculate_importance(self, content: str) -> float:
        """计算内容重要性分数"""
        # 关键词权重
        important_keywords = ['重要', '关键', '核心', '主要', '决定', '影响', '结果', '成功', '失败']
        keyword_score = sum(1 for word in important_keywords if word in content) / len(important_keywords)
        
        # 长度权重
        length_score = min(len(content) / 500, 1.0)
        
        # 情感强度权重
        emotional_score = abs(self._analyze_emotion(content)['valence'])
        
        return (keyword_score * 0.4 + length_score * 0.3 + emotional_score * 0.3)
    
    def _calculate_confidence(self, content: str, context: Dict[str, Any]) -> float:
        """计算内容置信度"""
        # 基于上下文信息完整性
        context_completeness = len(context) / 10  # 假设10个字段为完整
        
        # 基于内容清晰度
        clarity_score = 1.0 if len(content.split()) > 5 else 0.5
        
        return min(context_completeness * clarity_score, 1.0)
    
    def _analyze_emotion(self, content: str) -> Dict[str, float]:
        """分析情感特征"""
        # 简单情感词典分析
        positive_words = ['好', '棒', '优秀', '成功', '快乐', '满意', '积极']
        negative_words = ['坏', '差', '失败', '痛苦', '不满', '消极', '困难']
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        total_words = len(content.split())
        if total_words == 0:
            return {'valence': 0.0, 'arousal': 0.0}
        
        valence = (positive_count - negative_count) / max(total_words, 1)
        arousal = (positive_count + negative_count) / max(total_words, 1)
        
        return {
            'valence': max(-1, min(1, valence)),
            'arousal': max(0, min(1, arousal))
        }
    
    def _extract_tags(self, content: str) -> List[str]:
        """提取内容标签"""
        # 简单的关键词提取
        common_tags = ['反思', '学习', '改进', '分析', '总结', '经验', '教训', '发现']
        return [tag for tag in common_tags if tag in content]

class ReflectionPatternRecognizer:
    """反思模式识别器"""
    
    def __init__(self):
        self.pattern_models = {
            'temporal': self._recognize_temporal_patterns,
            'thematic': self._recognize_thematic_patterns,
            'behavioral': self._recognize_behavioral_patterns,
            'emotional': self._recognize_emotional_patterns
        }
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words=None)
        
    def recognize_patterns(self, contents: List[ReflectionContent]) -> List[ReflectionPattern]:
        """识别反思模式"""
        patterns = []
        
        for pattern_type, recognizer in self.pattern_models.items():
            try:
                type_patterns = recognizer(contents)
                patterns.extend(type_patterns)
            except Exception as e:
                logger.error(f"识别{pattern_type}模式失败: {e}")
        
        return patterns
    
    def _recognize_temporal_patterns(self, contents: List[ReflectionContent]) -> List[ReflectionPattern]:
        """识别时间模式"""
        patterns = []
        
        # 按时间排序
        sorted_contents = sorted(contents, key=lambda x: x.timestamp)
        
        # 识别周期性模式
        intervals = []
        for i in range(1, len(sorted_contents)):
            interval = (sorted_contents[i].timestamp - sorted_contents[i-1].timestamp).total_seconds()
            intervals.append(interval)
        
        if intervals:
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # 检查是否存在规律性
            if std_interval / avg_interval < 0.5:  # 变异系数小于0.5
                pattern_id = hashlib.md5(f"temporal_periodic_{avg_interval}".encode()).hexdigest()[:16]
                patterns.append(ReflectionPattern(
                    pattern_id=pattern_id,
                    pattern_type='temporal',
                    pattern_description=f"周期性反思模式，平均间隔{avg_interval/3600:.1f}小时",
                    frequency=len(intervals),
                    confidence=1.0 - (std_interval / avg_interval),
                    supporting_content_ids=[c.id for c in sorted_contents],
                    pattern_strength=1.0 - (std_interval / avg_interval),
                    first_observed=sorted_contents[0].timestamp,
                    last_observed=sorted_contents[-1].timestamp,
                    metadata={'avg_interval': avg_interval, 'std_interval': std_interval}
                ))
        
        return patterns
    
    def _recognize_thematic_patterns(self, contents: List[ReflectionContent]) -> List[ReflectionPattern]:
        """识别主题模式"""
        patterns = []
        
        # 提取文本内容
        texts = [content.content for content in contents]
        
        if len(texts) < 2:
            return patterns
        
        try:
            # TF-IDF向量化
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # K-means聚类
            n_clusters = min(5, len(texts))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(tfidf_matrix)
                
                # 计算轮廓系数
                silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
                
                # 为每个聚类创建模式
                feature_names = self.vectorizer.get_feature_names_out()
                for i in range(n_clusters):
                    cluster_contents = [contents[j] for j in range(len(contents)) if cluster_labels[j] == i]
                    
                    if len(cluster_contents) >= 2:
                        # 获取聚类中心特征词
                        cluster_center = kmeans.cluster_centers_[i]
                        top_features_idx = cluster_center.argsort()[-5:][::-1]
                        top_features = [feature_names[idx] for idx in top_features_idx]
                        
                        pattern_id = hashlib.md5(f"thematic_cluster_{i}".encode()).hexdigest()[:16]
                        patterns.append(ReflectionPattern(
                            pattern_id=pattern_id,
                            pattern_type='thematic',
                            pattern_description=f"主题聚类模式: {', '.join(top_features)}",
                            frequency=len(cluster_contents),
                            confidence=silhouette_avg,
                            supporting_content_ids=[c.id for c in cluster_contents],
                            pattern_strength=silhouette_avg,
                            first_observed=min(c.timestamp for c in cluster_contents),
                            last_observed=max(c.timestamp for c in cluster_contents),
                            metadata={'cluster_id': i, 'top_features': top_features}
                        ))
        except Exception as e:
            logger.error(f"主题模式识别失败: {e}")
        
        return patterns
    
    def _recognize_behavioral_patterns(self, contents: List[ReflectionContent]) -> List[ReflectionPattern]:
        """识别行为模式"""
        patterns = []
        
        # 分析行为序列
        action_sequences = []
        for content in contents:
            if content.content_type == 'action':
                action_sequences.append(content.content)
        
        if len(action_sequences) >= 3:
            # 寻找常见的行为序列
            sequence_counts = Counter()
            for i in range(len(action_sequences) - 2):
                sequence = ' -> '.join(action_sequences[i:i+3])
                sequence_counts[sequence] += 1
            
            # 找出频繁序列
            frequent_sequences = {seq: count for seq, count in sequence_counts.items() 
                                if count >= 2}
            
            for sequence, count in frequent_sequences.items():
                pattern_id = hashlib.md5(f"behavioral_{sequence}".encode()).hexdigest()[:16]
                patterns.append(ReflectionPattern(
                    pattern_id=pattern_id,
                    pattern_type='behavioral',
                    pattern_description=f"行为序列模式: {sequence}",
                    frequency=count,
                    confidence=count / len(action_sequences),
                    supporting_content_ids=[],
                    pattern_strength=count / len(action_sequences),
                    first_observed=datetime.datetime.now(),
                    last_observed=datetime.datetime.now(),
                    metadata={'sequence': sequence}
                ))
        
        return patterns
    
    def _recognize_emotional_patterns(self, contents: List[ReflectionContent]) -> List[ReflectionPattern]:
        """识别情感模式"""
        patterns = []
        
        # 分析情感趋势
        emotional_data = [(content.timestamp, content.emotional_valence, content.emotional_arousal) 
                         for content in contents if hasattr(content, 'emotional_valence')]
        
        if len(emotional_data) >= 3:
            valences = [data[1] for data in emotional_data]
            
            # 检查情感极性变化
            polarity_changes = 0
            for i in range(1, len(valences)):
                if valences[i] * valences[i-1] < 0:  # 符号变化
                    polarity_changes += 1
            
            if polarity_changes > 0:
                pattern_id = hashlib.md5("emotional_polarity".encode()).hexdigest()[:16]
                patterns.append(ReflectionPattern(
                    pattern_id=pattern_id,
                    pattern_type='emotional',
                    pattern_description=f"情感极性变化模式，共{polarity_changes}次变化",
                    frequency=polarity_changes,
                    confidence=min(polarity_changes / len(valences), 1.0),
                    supporting_content_ids=[c.id for c in contents],
                    pattern_strength=min(polarity_changes / len(valences), 1.0),
                    first_observed=min(c.timestamp for c in contents),
                    last_observed=max(c.timestamp for c in contents),
                    metadata={'polarity_changes': polarity_changes}
                ))
        
        return patterns

class ReflectionEvaluator:
    """反思结果评估器"""
    
    def __init__(self):
        self.evaluation_criteria = {
            'completeness': self._evaluate_completeness,
            'consistency': self._evaluate_consistency,
            'depth': self._evaluate_depth,
            'actionability': self._evaluate_actionability,
            'learning_value': self._evaluate_learning_value
        }
    
    def evaluate_reflection(self, content: ReflectionContent, patterns: List[ReflectionPattern]) -> Dict[str, float]:
        """评估反思质量"""
        scores = {}
        
        for criterion, evaluator in self.evaluation_criteria.items():
            try:
                scores[criterion] = evaluator(content, patterns)
            except Exception as e:
                logger.error(f"评估{criterion}失败: {e}")
                scores[criterion] = 0.0
        
        # 计算综合分数
        overall_score = np.mean(list(scores.values()))
        scores['overall'] = overall_score
        
        return scores
    
    def _evaluate_completeness(self, content: ReflectionContent, patterns: List[ReflectionPattern]) -> float:
        """评估完整性"""
        # 基于内容长度和上下文丰富度
        content_score = min(len(content.content) / 200, 1.0)  # 200字符为满分
        context_score = min(len(content.context) / 5, 1.0)   # 5个上下文字段为满分
        
        return (content_score + context_score) / 2
    
    def _evaluate_consistency(self, content: ReflectionContent, patterns: List[ReflectionPattern]) -> float:
        """评估一致性"""
        # 检查与现有模式的一致性
        if not patterns:
            return 0.5
        
        consistency_scores = []
        for pattern in patterns:
            if content.id in pattern.supporting_content_ids:
                consistency_scores.append(pattern.confidence)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _evaluate_depth(self, content: ReflectionContent, patterns: List[ReflectionPattern]) -> float:
        """评估深度"""
        # 基于内容复杂度和分析层次
        complexity_indicators = ['分析', '原因', '结果', '影响', '关系', '模式', '趋势']
        complexity_score = sum(1 for indicator in complexity_indicators 
                             if indicator in content.content) / len(complexity_indicators)
        
        # 基于反思层次
        reflection_levels = ['描述', '解释', '评估', '应用', '综合']
        level_score = sum(1 for level in reflection_levels 
                        if level in content.content) / len(reflection_levels)
        
        return (complexity_score + level_score) / 2
    
    def _evaluate_actionability(self, content: ReflectionContent, patterns: List[ReflectionPattern]) -> float:
        """评估可操作性"""
        action_indicators = ['应该', '需要', '建议', '改进', '调整', '改变', '行动']
        action_score = sum(1 for indicator in action_indicators 
                         if indicator in content.content) / len(action_indicators)
        
        return action_score
    
    def _evaluate_learning_value(self, content: ReflectionContent, patterns: List[ReflectionPattern]) -> float:
        """评估学习价值"""
        learning_indicators = ['学习', '理解', '发现', '认识', '掌握', '经验', '教训']
        learning_score = sum(1 for indicator in learning_indicators 
                           if indicator in content.content) / len(learning_indicators)
        
        # 基于重要性分数
        importance_score = content.importance_score
        
        return (learning_score + importance_score) / 2

class ReflectionValidator:
    """反思结果验证器"""
    
    def __init__(self):
        self.validation_methods = {
            'cross_reference': self._validate_cross_reference,
            'consistency_check': self._validate_consistency,
            'evidence_support': self._validate_evidence_support,
            'logical_coherence': self._validate_logical_coherence
        }
    
    def validate_insight(self, insight: ReflectionInsight, 
                        contents: List[ReflectionContent]) -> Tuple[bool, float]:
        """验证反思洞察"""
        validation_scores = []
        
        for method_name, validator in self.validation_methods.items():
            try:
                is_valid, score = validator(insight, contents)
                validation_scores.append(score)
            except Exception as e:
                logger.error(f"验证方法{method_name}失败: {e}")
                validation_scores.append(0.0)
        
        overall_score = np.mean(validation_scores)
        is_valid = overall_score >= 0.6  # 阈值设为0.6
        
        return is_valid, overall_score
    
    def _validate_cross_reference(self, insight: ReflectionInsight, 
                                 contents: List[ReflectionContent]) -> Tuple[bool, float]:
        """交叉引用验证"""
        # 检查洞察是否得到多个证据支持
        evidence_count = len(insight.supporting_evidence)
        min_evidence = 2
        
        if evidence_count >= min_evidence:
            return True, min(evidence_count / 5, 1.0)  # 最多5个证据为满分
        else:
            return False, evidence_count / min_evidence
    
    def _validate_consistency(self, insight: ReflectionInsight, 
                             contents: List[ReflectionContent]) -> Tuple[bool, float]:
        """一致性验证"""
        # 检查洞察内容的一致性
        consistency_indicators = ['一致', '符合', '匹配', '相符']
        consistency_score = sum(1 for indicator in consistency_indicators 
                              if indicator in insight.description) / len(consistency_indicators)
        
        return consistency_score > 0.3, consistency_score
    
    def _validate_evidence_support(self, insight: ReflectionInsight, 
                                  contents: List[ReflectionContent]) -> Tuple[bool, float]:
        """证据支持验证"""
        # 检查支持证据的质量
        if not insight.supporting_evidence:
            return False, 0.0
        
        evidence_scores = []
        for evidence_id in insight.supporting_evidence:
            # 找到对应的内容
            content = next((c for c in contents if c.id == evidence_id), None)
            if content:
                evidence_scores.append(content.confidence_score)
        
        if evidence_scores:
            avg_evidence_quality = np.mean(evidence_scores)
            return avg_evidence_quality > 0.5, avg_evidence_quality
        else:
            return False, 0.0
    
    def _validate_logical_coherence(self, insight: ReflectionInsight, 
                                   contents: List[ReflectionContent]) -> Tuple[bool, float]:
        """逻辑连贯性验证"""
        # 简单的逻辑检查
        logical_connectors = ['因为', '所以', '由于', '导致', '因此', '然而', '但是']
        coherence_score = sum(1 for connector in logical_connectors 
                            if connector in insight.description) / len(logical_connectors)
        
        return coherence_score > 0.2, coherence_score

class ReflectionExperienceExtractor:
    """反思经验提取器"""
    
    def __init__(self):
        self.extraction_strategies = {
            'success_patterns': self._extract_success_patterns,
            'failure_patterns': self._extract_failure_patterns,
            'learning_patterns': self._extract_learning_patterns,
            'adaptation_patterns': self._extract_adaptation_patterns
        }
    
    def extract_experiences(self, contents: List[ReflectionContent], 
                           patterns: List[ReflectionPattern]) -> List[ReflectionExperience]:
        """提取反思经验"""
        experiences = []
        
        for strategy_name, extractor in self.extraction_strategies.items():
            try:
                strategy_experiences = extractor(contents, patterns)
                experiences.extend(strategy_experiences)
            except Exception as e:
                logger.error(f"经验提取策略{strategy_name}失败: {e}")
        
        return experiences
    
    def _extract_success_patterns(self, contents: List[ReflectionContent], 
                                 patterns: List[ReflectionPattern]) -> List[ReflectionExperience]:
        """提取成功模式"""
        experiences = []
        
        # 识别成功相关内容
        success_contents = [c for c in contents if any(word in c.content 
                          for word in ['成功', '有效', '好', '优秀', '正确'])]
        
        if success_contents:
            # 提取共同特征
            common_features = self._find_common_features(success_contents)
            
            experience_id = hashlib.md5("success_pattern".encode()).hexdigest()[:16]
            experience = ReflectionExperience(
                experience_id=experience_id,
                experience_type='success',
                title='成功模式总结',
                description=f"基于{len(success_contents)}个成功案例总结的模式",
                context={'source_count': len(success_contents)},
                lessons_learned=common_features['lessons'],
                best_practices=common_features['practices'],
                pitfalls=[],
                applicable_scenarios=common_features['scenarios'],
                confidence_level=np.mean([c.confidence_score for c in success_contents]),
                created_at=datetime.datetime.now(),
                last_applied=None,
                usage_count=0,
                effectiveness_score=np.mean([c.importance_score for c in success_contents])
            )
            experiences.append(experience)
        
        return experiences
    
    def _extract_failure_patterns(self, contents: List[ReflectionContent], 
                                 patterns: List[ReflectionPattern]) -> List[ReflectionExperience]:
        """提取失败模式"""
        experiences = []
        
        # 识别失败相关内容
        failure_contents = [c for c in contents if any(word in c.content 
                          for word in ['失败', '错误', '问题', '困难', '不足'])]
        
        if failure_contents:
            common_features = self._find_common_features(failure_contents)
            
            experience_id = hashlib.md5("failure_pattern".encode()).hexdigest()[:16]
            experience = ReflectionExperience(
                experience_id=experience_id,
                experience_type='failure',
                title='失败模式总结',
                description=f"基于{len(failure_contents)}个失败案例总结的模式",
                context={'source_count': len(failure_contents)},
                lessons_learned=common_features['lessons'],
                best_practices=[],
                pitfalls=common_features['pitfalls'],
                applicable_scenarios=common_features['scenarios'],
                confidence_level=np.mean([c.confidence_score for c in failure_contents]),
                created_at=datetime.datetime.now(),
                last_applied=None,
                usage_count=0,
                effectiveness_score=1.0 - np.mean([c.importance_score for c in failure_contents])
            )
            experiences.append(experience)
        
        return experiences
    
    def _extract_learning_patterns(self, contents: List[ReflectionContent], 
                                  patterns: List[ReflectionPattern]) -> List[ReflectionExperience]:
        """提取学习模式"""
        experiences = []
        
        # 识别学习相关内容
        learning_contents = [c for c in contents if any(word in c.content 
                           for word in ['学习', '理解', '发现', '认识', '掌握'])]
        
        if learning_contents:
            common_features = self._find_common_features(learning_contents)
            
            experience_id = hashlib.md5("learning_pattern".encode()).hexdigest()[:16]
            experience = ReflectionExperience(
                experience_id=experience_id,
                experience_type='learning',
                title='学习模式总结',
                description=f"基于{len(learning_contents)}个学习案例总结的模式",
                context={'source_count': len(learning_contents)},
                lessons_learned=common_features['lessons'],
                best_practices=common_features['practices'],
                pitfalls=[],
                applicable_scenarios=common_features['scenarios'],
                confidence_level=np.mean([c.confidence_score for c in learning_contents]),
                created_at=datetime.datetime.now(),
                last_applied=None,
                usage_count=0,
                effectiveness_score=np.mean([c.importance_score for c in learning_contents])
            )
            experiences.append(experience)
        
        return experiences
    
    def _extract_adaptation_patterns(self, contents: List[ReflectionContent], 
                                    patterns: List[ReflectionPattern]) -> List[ReflectionExperience]:
        """提取适应模式"""
        experiences = []
        
        # 识别适应相关内容
        adaptation_contents = [c for c in contents if any(word in c.content 
                             for word in ['适应', '调整', '改变', '改进', '优化'])]
        
        if adaptation_contents:
            common_features = self._find_common_features(adaptation_contents)
            
            experience_id = hashlib.md5("adaptation_pattern".encode()).hexdigest()[:16]
            experience = ReflectionExperience(
                experience_id=experience_id,
                experience_type='adaptation',
                title='适应模式总结',
                description=f"基于{len(adaptation_contents)}个适应案例总结的模式",
                context={'source_count': len(adaptation_contents)},
                lessons_learned=common_features['lessons'],
                best_practices=common_features['practices'],
                pitfalls=[],
                applicable_scenarios=common_features['scenarios'],
                confidence_level=np.mean([c.confidence_score for c in adaptation_contents]),
                created_at=datetime.datetime.now(),
                last_applied=None,
                usage_count=0,
                effectiveness_score=np.mean([c.importance_score for c in adaptation_contents])
            )
            experiences.append(experience)
        
        return experiences
    
    def _find_common_features(self, contents: List[ReflectionContent]) -> Dict[str, List[str]]:
        """寻找共同特征"""
        # 简单的特征提取
        lessons = []
        practices = []
        pitfalls = []
        scenarios = []
        
        for content in contents:
            if '学习到' in content.content or '认识到' in content.content:
                lessons.append(content.content[:50] + '...' if len(content.content) > 50 else content.content)
            
            if '应该' in content.content or '需要' in content.content:
                practices.append(content.content[:50] + '...' if len(content.content) > 50 else content.content)
            
            if '避免' in content.content or '不要' in content.content:
                pitfalls.append(content.content[:50] + '...' if len(content.content) > 50 else content.content)
            
            if '在' in content.content and '情况下' in content.content:
                scenarios.append(content.content[:50] + '...' if len(content.content) > 50 else content.content)
        
        return {
            'lessons': lessons[:3],  # 最多3个
            'practices': practices[:3],
            'pitfalls': pitfalls[:3],
            'scenarios': scenarios[:3]
        }

class ReflectionEffectTracker:
    """反思效果跟踪器"""
    
    def __init__(self):
        self.tracking_metrics = {
            'frequency': self._track_frequency,
            'quality': self._track_quality,
            'impact': self._track_impact,
            'learning_rate': self._track_learning_rate
        }
        self.effect_history = []
    
    def track_effect(self, reflection_id: str, actions_taken: List[str], 
                    outcomes: List[str]) -> Dict[str, float]:
        """跟踪反思效果"""
        metrics = {}
        
        for metric_name, tracker in self.tracking_metrics.items():
            try:
                metrics[metric_name] = tracker(reflection_id, actions_taken, outcomes)
            except Exception as e:
                logger.error(f"跟踪指标{metric_name}失败: {e}")
                metrics[metric_name] = 0.0
        
        # 记录到历史
        self.effect_history.append({
            'reflection_id': reflection_id,
            'timestamp': datetime.datetime.now(),
            'metrics': metrics,
            'actions': actions_taken,
            'outcomes': outcomes
        })
        
        return metrics
    
    def _track_frequency(self, reflection_id: str, actions_taken: List[str], 
                        outcomes: List[str]) -> float:
        """跟踪行动频率"""
        return len(actions_taken) / 10.0  # 标准化到0-1
    
    def _track_quality(self, reflection_id: str, actions_taken: List[str], 
                      outcomes: List[str]) -> float:
        """跟踪行动质量"""
        if not outcomes:
            return 0.0
        
        positive_outcomes = sum(1 for outcome in outcomes if any(word in outcome 
                               for word in ['成功', '有效', '改善', '进步']))
        return positive_outcomes / len(outcomes)
    
    def _track_impact(self, reflection_id: str, actions_taken: List[str], 
                     outcomes: List[str]) -> float:
        """跟踪影响程度"""
        impact_indicators = ['重大', '显著', '明显', '重要', '关键']
        impact_score = sum(1 for outcome in outcomes 
                          for indicator in impact_indicators 
                          if indicator in outcome)
        return min(impact_score / 5.0, 1.0)
    
    def _track_learning_rate(self, reflection_id: str, actions_taken: List[str], 
                            outcomes: List[str]) -> float:
        """跟踪学习速度"""
        learning_indicators = ['学会', '掌握', '理解', '发现', '认识']
        learning_score = sum(1 for action in actions_taken 
                           for indicator in learning_indicators 
                           if indicator in action)
        return min(learning_score / 3.0, 1.0)
    
    def get_effect_trend(self, reflection_id: str, days: int = 30) -> Dict[str, List[float]]:
        """获取效果趋势"""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        recent_history = [h for h in self.effect_history 
                         if h['timestamp'] >= cutoff_date and h['reflection_id'] == reflection_id]
        
        if not recent_history:
            return {}
        
        trends = {}
        for metric in self.tracking_metrics.keys():
            trends[metric] = [h['metrics'][metric] for h in recent_history]
        
        return trends

class ReflectionHistoryManager:
    """反思历史管理器"""
    
    def __init__(self, db_path: str = "reflection_history.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建反思内容表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reflection_contents (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                content_type TEXT,
                content TEXT,
                context TEXT,
                tags TEXT,
                importance_score REAL,
                confidence_score REAL,
                emotional_valence REAL,
                emotional_arousal REAL,
                source TEXT,
                metadata TEXT
            )
        ''')
        
        # 创建反思模式表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reflection_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                pattern_description TEXT,
                frequency INTEGER,
                confidence REAL,
                supporting_content_ids TEXT,
                pattern_strength REAL,
                first_observed TEXT,
                last_observed TEXT,
                metadata TEXT
            )
        ''')
        
        # 创建反思洞察表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reflection_insights (
                insight_id TEXT PRIMARY KEY,
                insight_type TEXT,
                title TEXT,
                description TEXT,
                supporting_evidence TEXT,
                confidence_score REAL,
                impact_score REAL,
                actionable BOOLEAN,
                recommendations TEXT,
                created_at TEXT,
                validated BOOLEAN,
                validation_score REAL
            )
        ''')
        
        # 创建反思经验表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reflection_experiences (
                experience_id TEXT PRIMARY KEY,
                experience_type TEXT,
                title TEXT,
                description TEXT,
                context TEXT,
                lessons_learned TEXT,
                best_practices TEXT,
                pitfalls TEXT,
                applicable_scenarios TEXT,
                confidence_level REAL,
                created_at TEXT,
                last_applied TEXT,
                usage_count INTEGER,
                effectiveness_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_content(self, content: ReflectionContent):
        """存储反思内容"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO reflection_contents 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            content.id, content.timestamp.isoformat(), content.content_type,
            content.content, json.dumps(content.context, ensure_ascii=False),
            json.dumps(content.tags, ensure_ascii=False), content.importance_score,
            content.confidence_score, content.emotional_valence, content.emotional_arousal,
            content.source, json.dumps(content.metadata, ensure_ascii=False)
        ))
        
        conn.commit()
        conn.close()
    
    def store_pattern(self, pattern: ReflectionPattern):
        """存储反思模式"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO reflection_patterns 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pattern_id, pattern.pattern_type, pattern.pattern_description,
            pattern.frequency, pattern.confidence,
            json.dumps(pattern.supporting_content_ids, ensure_ascii=False),
            pattern.pattern_strength, pattern.first_observed.isoformat(),
            pattern.last_observed.isoformat(), json.dumps(pattern.metadata, ensure_ascii=False)
        ))
        
        conn.commit()
        conn.close()
    
    def store_insight(self, insight: ReflectionInsight):
        """存储反思洞察"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO reflection_insights 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            insight.insight_id, insight.insight_type, insight.title,
            insight.description, json.dumps(insight.supporting_evidence, ensure_ascii=False),
            insight.confidence_score, insight.impact_score, insight.actionable,
            json.dumps(insight.recommendations, ensure_ascii=False),
            insight.created_at.isoformat(), insight.validated, insight.validation_score
        ))
        
        conn.commit()
        conn.close()
    
    def store_experience(self, experience: ReflectionExperience):
        """存储反思经验"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO reflection_experiences 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experience.experience_id, experience.experience_type, experience.title,
            experience.description, json.dumps(experience.context, ensure_ascii=False),
            json.dumps(experience.lessons_learned, ensure_ascii=False),
            json.dumps(experience.best_practices, ensure_ascii=False),
            json.dumps(experience.pitfalls, ensure_ascii=False),
            json.dumps(experience.applicable_scenarios, ensure_ascii=False),
            experience.confidence_level, experience.created_at.isoformat(),
            experience.last_applied.isoformat() if experience.last_applied else None,
            experience.usage_count, experience.effectiveness_score
        ))
        
        conn.commit()
        conn.close()
    
    def query_contents(self, filters: Dict[str, Any] = None) -> List[ReflectionContent]:
        """查询反思内容"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM reflection_contents WHERE 1=1"
        params = []
        
        if filters:
            if 'content_type' in filters:
                query += " AND content_type = ?"
                params.append(filters['content_type'])
            
            if 'source' in filters:
                query += " AND source = ?"
                params.append(filters['source'])
            
            if 'start_date' in filters:
                query += " AND timestamp >= ?"
                params.append(filters['start_date'])
            
            if 'end_date' in filters:
                query += " AND timestamp <= ?"
                params.append(filters['end_date'])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        contents = []
        for row in rows:
            content = ReflectionContent(
                id=row[0], timestamp=datetime.datetime.fromisoformat(row[1]),
                content_type=row[2], content=row[3],
                context=json.loads(row[4]), tags=json.loads(row[5]),
                importance_score=row[6], confidence_score=row[7],
                emotional_valence=row[8], emotional_arousal=row[9],
                source=row[10], metadata=json.loads(row[11])
            )
            contents.append(content)
        
        return contents
    
    def query_patterns(self, pattern_type: str = None) -> List[ReflectionPattern]:
        """查询反思模式"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if pattern_type:
            cursor.execute("SELECT * FROM reflection_patterns WHERE pattern_type = ?", (pattern_type,))
        else:
            cursor.execute("SELECT * FROM reflection_patterns")
        
        rows = cursor.fetchall()
        conn.close()
        
        patterns = []
        for row in rows:
            pattern = ReflectionPattern(
                pattern_id=row[0], pattern_type=row[1], pattern_description=row[2],
                frequency=row[3], confidence=row[4],
                supporting_content_ids=json.loads(row[5]), pattern_strength=row[6],
                first_observed=datetime.datetime.fromisoformat(row[7]),
                last_observed=datetime.datetime.fromisoformat(row[8]),
                metadata=json.loads(row[9])
            )
            patterns.append(pattern)
        
        return patterns
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 内容统计
        cursor.execute("SELECT COUNT(*) FROM reflection_contents")
        content_count = cursor.fetchone()[0]
        
        # 模式统计
        cursor.execute("SELECT COUNT(*) FROM reflection_patterns")
        pattern_count = cursor.fetchone()[0]
        
        # 洞察统计
        cursor.execute("SELECT COUNT(*) FROM reflection_insights")
        insight_count = cursor.fetchone()[0]
        
        # 经验统计
        cursor.execute("SELECT COUNT(*) FROM reflection_experiences")
        experience_count = cursor.fetchone()[0]
        
        # 时间范围
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM reflection_contents")
        min_date, max_date = cursor.fetchone()
        
        conn.close()
        
        return {
            'content_count': content_count,
            'pattern_count': pattern_count,
            'insight_count': insight_count,
            'experience_count': experience_count,
            'date_range': (min_date, max_date) if min_date else None
        }

class ReflectionReportGenerator:
    """反思报告生成器"""
    
    def __init__(self):
        self.report_templates = {
            'summary': self._generate_summary_report,
            'detailed': self._generate_detailed_report,
            'trends': self._generate_trends_report,
            'insights': self._generate_insights_report
        }
    
    def generate_report(self, report_type: str, contents: List[ReflectionContent],
                       patterns: List[ReflectionPattern], insights: List[ReflectionInsight],
                       experiences: List[ReflectionExperience]) -> str:
        """生成反思报告"""
        generator = self.report_templates.get(report_type, self._generate_summary_report)
        return generator(contents, patterns, insights, experiences)
    
    def _generate_summary_report(self, contents: List[ReflectionContent],
                                patterns: List[ReflectionPattern], insights: List[ReflectionInsight],
                                experiences: List[ReflectionExperience]) -> str:
        """生成总结报告"""
        report = []
        report.append("# 深度反思总结报告")
        report.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 基本统计
        report.append("## 基本统计")
        report.append(f"- 反思内容数量: {len(contents)}")
        report.append(f"- 识别模式数量: {len(patterns)}")
        report.append(f"- 生成洞察数量: {len(insights)}")
        report.append(f"- 提取经验数量: {len(experiences)}")
        report.append("")
        
        # 模式分析
        if patterns:
            report.append("## 主要模式")
            for pattern in patterns[:5]:  # 显示前5个模式
                report.append(f"- **{pattern.pattern_type}**: {pattern.pattern_description}")
                report.append(f"  - 置信度: {pattern.confidence:.2f}")
                report.append(f"  - 支持内容数: {pattern.frequency}")
            report.append("")
        
        # 关键洞察
        if insights:
            report.append("## 关键洞察")
            high_confidence_insights = [i for i in insights if i.confidence_score > 0.7]
            for insight in high_confidence_insights[:3]:  # 显示前3个高置信度洞察
                report.append(f"- **{insight.title}**: {insight.description}")
                report.append(f"  - 置信度: {insight.confidence_score:.2f}")
            report.append("")
        
        # 重要经验
        if experiences:
            report.append("## 重要经验")
            effective_experiences = sorted(experiences, key=lambda x: x.effectiveness_score, reverse=True)
            for experience in effective_experiences[:3]:  # 显示前3个高效经验
                report.append(f"- **{experience.title}**: {experience.description}")
                report.append(f"  - 效果评分: {experience.effectiveness_score:.2f}")
                report.append(f"  - 使用次数: {experience.usage_count}")
            report.append("")
        
        # 建议和下一步行动
        report.append("## 建议和下一步行动")
        report.append("基于本次深度反思分析，建议关注以下方面:")
        report.append("1. 继续收集反思内容，保持反思习惯")
        report.append("2. 重点关注识别出的模式和洞察")
        report.append("3. 应用提取的经验到实际场景中")
        report.append("4. 定期回顾和更新反思记录")
        
        return "\n".join(report)
    
    def _generate_detailed_report(self, contents: List[ReflectionContent],
                                 patterns: List[ReflectionPattern], insights: List[ReflectionInsight],
                                 experiences: List[ReflectionExperience]) -> str:
        """生成详细报告"""
        report = []
        report.append("# 深度反思详细报告")
        report.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 详细内容分析
        report.append("## 反思内容详细分析")
        for i, content in enumerate(contents, 1):
            report.append(f"### 内容 {i}")
            report.append(f"**时间**: {content.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"**类型**: {content.content_type}")
            report.append(f"**内容**: {content.content}")
            report.append(f"**重要性**: {content.importance_score:.2f}")
            report.append(f"**置信度**: {content.confidence_score:.2f}")
            report.append(f"**情感极性**: {content.emotional_valence:.2f}")
            report.append("")
        
        # 模式详细分析
        report.append("## 模式详细分析")
        for pattern in patterns:
            report.append(f"### {pattern.pattern_type}模式")
            report.append(f"**描述**: {pattern.pattern_description}")
            report.append(f"**频率**: {pattern.frequency}")
            report.append(f"**置信度**: {pattern.confidence:.2f}")
            report.append(f"**强度**: {pattern.pattern_strength:.2f}")
            report.append(f"**时间范围**: {pattern.first_observed.strftime('%Y-%m-%d')} - {pattern.last_observed.strftime('%Y-%m-%d')}")
            report.append("")
        
        return "\n".join(report)
    
    def _generate_trends_report(self, contents: List[ReflectionContent],
                               patterns: List[ReflectionPattern], insights: List[ReflectionInsight],
                               experiences: List[ReflectionExperience]) -> str:
        """生成趋势报告"""
        report = []
        report.append("# 反思趋势分析报告")
        report.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 时间趋势分析
        if contents:
            # 按时间排序
            sorted_contents = sorted(contents, key=lambda x: x.timestamp)
            
            # 分析重要性趋势
            importance_trend = [c.importance_score for c in sorted_contents]
            
            report.append("## 重要性趋势")
            if len(importance_trend) > 1:
                trend_direction = "上升" if importance_trend[-1] > importance_trend[0] else "下降"
                report.append(f"反思重要性整体呈{trend_direction}趋势")
                report.append(f"起始重要性: {importance_trend[0]:.2f}")
                report.append(f"当前重要性: {importance_trend[-1]:.2f}")
            report.append("")
        
        # 模式演进
        if patterns:
            report.append("## 模式演进")
            temporal_patterns = [p for p in patterns if p.pattern_type == 'temporal']
            if temporal_patterns:
                report.append("发现以下时间模式:")
                for pattern in temporal_patterns:
                    report.append(f"- {pattern.pattern_description}")
            report.append("")
        
        return "\n".join(report)
    
    def _generate_insights_report(self, contents: List[ReflectionContent],
                                 patterns: List[ReflectionPattern], insights: List[ReflectionInsight],
                                 experiences: List[ReflectionExperience]) -> str:
        """生成洞察报告"""
        report = []
        report.append("# 反思洞察报告")
        report.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 洞察分类
        insight_types = {}
        for insight in insights:
            if insight.insight_type not in insight_types:
                insight_types[insight.insight_type] = []
            insight_types[insight.insight_type].append(insight)
        
        for insight_type, type_insights in insight_types.items():
            report.append(f"## {insight_type}洞察")
            for insight in type_insights:
                report.append(f"### {insight.title}")
                report.append(f"**描述**: {insight.description}")
                report.append(f"**置信度**: {insight.confidence_score:.2f}")
                report.append(f"**影响度**: {insight.impact_score:.2f}")
                report.append(f"**可操作性**: {'是' if insight.actionable else '否'}")
                if insight.recommendations:
                    report.append("**建议**:")
                    for rec in insight.recommendations:
                        report.append(f"- {rec}")
                report.append("")
        
        return "\n".join(report)

class DeepReflection:
    """深度反思器主类"""
    
    def __init__(self, db_path: str = "deep_reflection.db"):
        """初始化深度反思器"""
        self.content_collector = ReflectionContentCollector()
        self.pattern_recognizer = ReflectionPatternRecognizer()
        self.evaluator = ReflectionEvaluator()
        self.validator = ReflectionValidator()
        self.experience_extractor = ReflectionExperienceExtractor()
        self.effect_tracker = ReflectionEffectTracker()
        self.history_manager = ReflectionHistoryManager(db_path)
        self.report_generator = ReflectionReportGenerator()
        
        self.contents = []
        self.patterns = []
        self.insights = []
        self.experiences = []
        
        logger.info("深度反思器初始化完成")
    
    def reflect(self, content_type: str, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行深度反思流程"""
        try:
            logger.info(f"开始反思流程: {content_type}")
            
            # 1. 收集反思内容
            reflection_content = self.content_collector.collect_content(
                content_type, 'manual', content, context or {}
            )
            self.contents.append(reflection_content)
            self.history_manager.store_content(reflection_content)
            
            # 2. 识别反思模式
            if len(self.contents) >= 2:  # 至少需要2个内容才能识别模式
                self.patterns = self.pattern_recognizer.recognize_patterns(self.contents)
                for pattern in self.patterns:
                    self.history_manager.store_pattern(pattern)
            
            # 3. 评估反思质量
            evaluation_scores = self.evaluator.evaluate_reflection(reflection_content, self.patterns)
            
            # 4. 生成反思洞察
            new_insights = self._generate_insights(reflection_content)
            self.insights.extend(new_insights)
            
            # 5. 提取反思经验
            if len(self.contents) >= 5:  # 至少需要5个内容才能提取经验
                new_experiences = self.experience_extractor.extract_experiences(self.contents, self.patterns)
                self.experiences.extend(new_experiences)
                for experience in new_experiences:
                    self.history_manager.store_experience(experience)
            
            # 6. 验证重要洞察
            validated_insights = []
            for insight in new_insights:
                is_valid, validation_score = self.validator.validate_insight(insight, self.contents)
                if is_valid:
                    insight.validated = True
                    insight.validation_score = validation_score
                    validated_insights.append(insight)
                    self.history_manager.store_insight(insight)
            
            result = {
                'content': reflection_content,
                'patterns': self.patterns,
                'insights': validated_insights,
                'experiences': self.experiences,
                'evaluation': evaluation_scores,
                'summary': self._generate_reflection_summary(reflection_content, evaluation_scores)
            }
            
            logger.info("反思流程完成")
            return result
            
        except Exception as e:
            logger.error(f"反思流程失败: {e}")
            raise
    
    def _generate_insights(self, content: ReflectionContent) -> List[ReflectionInsight]:
        """生成反思洞察"""
        insights = []
        
        # 基于内容特征生成洞察
        if content.importance_score > 0.7:
            insight_id = hashlib.md5(f"importance_{content.id}".encode()).hexdigest()[:16]
            insights.append(ReflectionInsight(
                insight_id=insight_id,
                insight_type='importance',
                title='高重要性内容洞察',
                description=f'该反思内容重要性评分为{content.importance_score:.2f}，值得重点关注',
                supporting_evidence=[content.id],
                confidence_score=content.importance_score,
                impact_score=content.importance_score * 0.8,
                actionable=True,
                recommendations=['深入分析', '制定行动计划', '定期回顾'],
                created_at=datetime.datetime.now(),
                validated=False,
                validation_score=0.0
            ))
        
        # 基于情感特征生成洞察
        if abs(content.emotional_valence) > 0.5:
            emotion_type = 'positive' if content.emotional_valence > 0 else 'negative'
            insight_id = hashlib.md5(f"emotion_{content.id}".encode()).hexdigest()[:16]
            insights.append(ReflectionInsight(
                insight_id=insight_id,
                insight_type='emotional',
                title=f'{emotion_type}情感洞察',
                description=f'检测到{emotion_type}情感倾向，情感极性为{content.emotional_valence:.2f}',
                supporting_evidence=[content.id],
                confidence_score=abs(content.emotional_valence),
                impact_score=content.emotional_arousal,
                actionable=True,
                recommendations=['情感调节', '心态调整', '情绪管理'] if emotion_type == 'negative' else ['保持状态', '分享经验'],
                created_at=datetime.datetime.now(),
                validated=False,
                validation_score=0.0
            ))
        
        return insights
    
    def _generate_reflection_summary(self, content: ReflectionContent, evaluation: Dict[str, float]) -> str:
        """生成反思总结"""
        summary_parts = []
        
        summary_parts.append(f"本次反思内容重要性评分为{content.importance_score:.2f}")
        summary_parts.append(f"综合质量评分为{evaluation['overall']:.2f}")
        
        if evaluation['overall'] > 0.7:
            summary_parts.append("反思质量较高，建议继续深入分析")
        elif evaluation['overall'] > 0.4:
            summary_parts.append("反思质量中等，建议提高分析的深度和广度")
        else:
            summary_parts.append("反思质量有待提高，建议增加更多背景信息和分析")
        
        return "；".join(summary_parts) + "。"
    
    def track_effect(self, reflection_id: str, actions: List[str], outcomes: List[str]) -> Dict[str, float]:
        """跟踪反思效果"""
        return self.effect_tracker.track_effect(reflection_id, actions, outcomes)
    
    def query_history(self, filters: Dict[str, Any] = None) -> List[ReflectionContent]:
        """查询反思历史"""
        return self.history_manager.query_contents(filters)
    
    def generate_report(self, report_type: str = 'summary') -> str:
        """生成反思报告"""
        return self.report_generator.generate_report(
            report_type, self.contents, self.patterns, self.insights, self.experiences
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.history_manager.get_statistics()
    
    def export_data(self, filepath: str):
        """导出反思数据"""
        data = {
            'contents': [asdict(c) for c in self.contents],
            'patterns': [asdict(p) for p in self.patterns],
            'insights': [asdict(i) for i in self.insights],
            'experiences': [asdict(e) for e in self.experiences],
            'export_time': datetime.datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"数据已导出到: {filepath}")
    
    def import_data(self, filepath: str):
        """导入反思数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建对象
        self.contents = [ReflectionContent(**item) for item in data.get('contents', [])]
        self.patterns = [ReflectionPattern(**item) for item in data.get('patterns', [])]
        self.insights = [ReflectionInsight(**item) for item in data.get('insights', [])]
        self.experiences = [ReflectionExperience(**item) for item in data.get('experiences', [])]
        
        logger.info(f"数据已从{filepath}导入")

def main():
    """主函数 - 演示深度反思器功能"""
    print("=== H1深度反思器演示 ===\n")
    
    # 创建深度反思器实例
    reflector = DeepReflection()
    
    # 模拟反思过程
    reflection_scenarios = [
        {
            'type': 'thought',
            'content': '今天的工作效率比昨天提高了，主要是因为采用了新的时间管理方法',
            'context': {'workload': 'normal', 'mood': 'positive'}
        },
        {
            'type': 'action',
            'content': '实施了番茄工作法，每25分钟休息5分钟，效果显著',
            'context': {'method': 'pomodoro', 'duration': '2hours'}
        },
        {
            'type': 'result',
            'content': '任务完成质量提升，错误率下降，但需要避免过度休息',
            'context': {'quality': 'improved', 'errors': 'reduced'}
        },
        {
            'type': 'emotion',
            'content': '对新的工作方法感到满意和自信，但有时会担心效果不持久',
            'context': {'satisfaction': 'high', 'confidence': 'medium'}
        },
        {
            'type': 'learning',
            'content': '学会了时间管理的重要性，发现专注度和休息时间的平衡很关键',
            'context': {'learning': 'time_management', 'balance': 'important'}
        }
    ]
    
    # 执行反思
    results = []
    for scenario in reflection_scenarios:
        print(f"反思场景: {scenario['type']}")
        print(f"内容: {scenario['content']}")
        
        result = reflector.reflect(
            scenario['type'], 
            scenario['content'], 
            scenario['context']
        )
        
        results.append(result)
        print(f"质量评分: {result['evaluation']['overall']:.2f}")
        print(f"总结: {result['summary']}")
        print("-" * 50)
    
    # 生成报告
    print("\n=== 生成反思报告 ===")
    report = reflector.generate_report('summary')
    print(report)
    
    # 显示统计信息
    print("\n=== 统计信息 ===")
    stats = reflector.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 跟踪效果
    print("\n=== 效果跟踪演示 ===")
    if results:
        first_result = results[0]
        effect_metrics = reflector.track_effect(
            first_result['content'].id,
            ['采用新方法', '调整时间安排', '增加休息频率'],
            ['效率提升', '错误减少', '满意度提高']
        )
        print("效果指标:", effect_metrics)
    
    print("\n=== 演示完成 ===")

if __name__ == "__main__":
    main()