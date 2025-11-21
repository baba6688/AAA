#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C8知识验证器
实现知识一致性检验、完整性检查、准确性验证、时效性评估、冲突检测、质量评分和修复优化功能


版本: 1.0.0
创建时间: 2025-11-05
"""

import json
import logging
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """验证级别枚举"""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    COMPREHENSIVE = "comprehensive"


class QualityScore(Enum):
    """质量评分等级"""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 80-89
    FAIR = "fair"           # 70-79
    POOR = "poor"           # 60-69
    CRITICAL = "critical"   # 0-59


@dataclass
class ValidationResult:
    """验证结果数据类"""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeQualityReport:
    """知识质量报告"""
    overall_score: float
    quality_level: QualityScore
    validation_results: List[ValidationResult]
    summary: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)


class KnowledgeValidator:
    """知识验证器主类"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        初始化知识验证器
        
        Args:
            validation_level: 验证级别
        """
        self.validation_level = validation_level
        self.validation_history = []
        self.quality_metrics = {}
        self.conflict_patterns = []
        self.accuracy_baselines = {}
        
        # 初始化验证规则
        self._init_validation_rules()
        
        logger.info(f"知识验证器已初始化，验证级别: {validation_level.value}")
    
    def _init_validation_rules(self):
        """初始化验证规则"""
        # 知识完整性规则
        self.completeness_rules = {
            'required_fields': ['id', 'content', 'timestamp', 'source'],
            'min_content_length': 10,
            'max_content_length': 10000,
            'required_metadata': ['confidence', 'category']
        }
        
        # 知识一致性规则
        self.consistency_rules = {
            'max_contradiction_score': 0.3,
            'min_coherence_score': 0.7,
            'temporal_consistency_window': 30  # 天
        }
        
        # 知识准确性规则
        self.accuracy_rules = {
            'min_source_reliability': 0.6,
            'cross_validation_threshold': 0.8,
            'expert_validation_required': False
        }
        
        # 知识时效性规则
        self.timeliness_rules = {
            'max_age_days': 365,
            'update_frequency_threshold': 7,  # 天
            'decay_rate': 0.1
        }
    
    def validate_knowledge_base(self, knowledge_items: List[Dict[str, Any]]) -> KnowledgeQualityReport:
        """
        验证整个知识库
        
        Args:
            knowledge_items: 知识项列表
            
        Returns:
            知识质量报告
        """
        logger.info(f"开始验证知识库，包含 {len(knowledge_items)} 个知识项")
        
        validation_results = []
        
        # 1. 知识一致性检验
        consistency_result = self._validate_consistency(knowledge_items)
        validation_results.append(consistency_result)
        
        # 2. 知识完整性检查
        completeness_result = self._validate_completeness(knowledge_items)
        validation_results.append(completeness_result)
        
        # 3. 知识准确性验证
        accuracy_result = self._validate_accuracy(knowledge_items)
        validation_results.append(accuracy_result)
        
        # 4. 知识时效性评估
        timeliness_result = self._validate_timeliness(knowledge_items)
        validation_results.append(timeliness_result)
        
        # 5. 知识冲突检测
        conflict_result = self._detect_conflicts(knowledge_items)
        validation_results.append(conflict_result)
        
        # 6. 知识质量评分
        quality_result = self._calculate_quality_score(knowledge_items, validation_results)
        validation_results.append(quality_result)
        
        # 7. 生成修复建议
        repair_suggestions = self._generate_repair_suggestions(validation_results)
        
        # 计算总体质量分数
        overall_score = self._calculate_overall_score(validation_results)
        quality_level = self._determine_quality_level(overall_score)
        
        # 生成报告摘要
        summary = self._generate_summary(validation_results, knowledge_items)
        
        # 创建质量报告
        report = KnowledgeQualityReport(
            overall_score=overall_score,
            quality_level=quality_level,
            validation_results=validation_results,
            summary=summary,
            recommendations=repair_suggestions
        )
        
        # 保存验证历史
        self.validation_history.append(report)
        
        logger.info(f"知识库验证完成，总体质量分数: {overall_score:.2f}")
        
        return report
    
    def _validate_consistency(self, knowledge_items: List[Dict[str, Any]]) -> ValidationResult:
        """验证知识一致性"""
        logger.info("执行知识一致性检验")
        
        contradictions = []
        coherence_scores = []
        
        # 检查逻辑矛盾
        for i, item1 in enumerate(knowledge_items):
            for j, item2 in enumerate(knowledge_items[i+1:], i+1):
                if self._check_logical_contradiction(item1, item2):
                    contradictions.append({
                        'item1_id': item1.get('id', i),
                        'item2_id': item2.get('id', j),
                        'contradiction_type': 'logical'
                    })
                
                # 计算连贯性分数
                coherence = self._calculate_coherence(item1, item2)
                coherence_scores.append(coherence)
        
        # 计算一致性指标
        contradiction_score = len(contradictions) / max(len(knowledge_items), 1)
        avg_coherence = statistics.mean(coherence_scores) if coherence_scores else 1.0
        
        # 一致性分数 (0-1, 越高越好)
        consistency_score = max(0, 1 - contradiction_score) * avg_coherence
        
        passed = (contradiction_score <= self.consistency_rules['max_contradiction_score'] and 
                 avg_coherence >= self.consistency_rules['min_coherence_score'])
        
        suggestions = []
        if contradiction_score > self.consistency_rules['max_contradiction_score']:
            suggestions.append("发现逻辑矛盾，需要进一步核实相关信息")
        if avg_coherence < self.consistency_rules['min_coherence_score']:
            suggestions.append("知识项之间的连贯性不足，建议加强逻辑关联")
        
        return ValidationResult(
            test_name="知识一致性检验",
            passed=passed,
            score=consistency_score,
            details={
                'contradictions_found': len(contradictions),
                'contradiction_score': contradiction_score,
                'average_coherence': avg_coherence,
                'contradiction_details': contradictions
            },
            suggestions=suggestions
        )
    
    def _validate_completeness(self, knowledge_items: List[Dict[str, Any]]) -> ValidationResult:
        """验证知识完整性"""
        logger.info("执行知识完整性检查")
        
        incomplete_items = []
        missing_fields = defaultdict(int)
        
        for item in knowledge_items:
            item_missing = []
            
            # 检查必需字段
            for field in self.completeness_rules['required_fields']:
                if field not in item or not item[field]:
                    item_missing.append(field)
                    missing_fields[field] += 1
            
            # 检查内容长度
            content = item.get('content', '')
            if len(content) < self.completeness_rules['min_content_length']:
                item_missing.append('content_too_short')
                missing_fields['content_too_short'] += 1
            elif len(content) > self.completeness_rules['max_content_length']:
                item_missing.append('content_too_long')
                missing_fields['content_too_long'] += 1
            
            # 检查元数据
            for field in self.completeness_rules['required_metadata']:
                if field not in item or item[field] is None:
                    item_missing.append(field)
                    missing_fields[field] += 1
            
            if item_missing:
                incomplete_items.append({
                    'item_id': item.get('id'),
                    'missing_fields': item_missing
                })
        
        # 计算完整性分数
        completeness_score = 1 - (len(incomplete_items) / max(len(knowledge_items), 1))
        
        passed = completeness_score >= 0.8
        
        suggestions = []
        if incomplete_items:
            suggestions.append(f"发现 {len(incomplete_items)} 个不完整的知识项，需要补充缺失信息")
            top_missing = sorted(missing_fields.items(), key=lambda x: x[1], reverse=True)[:3]
            for field, count in top_missing:
                suggestions.append(f"建议补充 '{field}' 字段，缺失次数: {count}")
        
        return ValidationResult(
            test_name="知识完整性检查",
            passed=passed,
            score=completeness_score,
            details={
                'total_items': len(knowledge_items),
                'incomplete_items': len(incomplete_items),
                'completeness_rate': completeness_score,
                'missing_fields': dict(missing_fields),
                'incomplete_details': incomplete_items
            },
            suggestions=suggestions
        )
    
    def _validate_accuracy(self, knowledge_items: List[Dict[str, Any]]) -> ValidationResult:
        """验证知识准确性"""
        logger.info("执行知识准确性验证")
        
        accuracy_scores = []
        validation_failures = []
        
        for item in knowledge_items:
            # 评估来源可靠性
            source_reliability = self._evaluate_source_reliability(item.get('source', ''))
            
            # 交叉验证
            cross_validation_score = self._perform_cross_validation(item, knowledge_items)
            
            # 计算准确性分数
            accuracy = (source_reliability * 0.4 + cross_validation_score * 0.6)
            accuracy_scores.append(accuracy)
            
            # 检查是否需要专家验证
            if (accuracy < self.accuracy_rules['cross_validation_threshold'] or 
                source_reliability < self.accuracy_rules['min_source_reliability']):
                validation_failures.append({
                    'item_id': item.get('id'),
                    'accuracy': accuracy,
                    'source_reliability': source_reliability,
                    'cross_validation': cross_validation_score
                })
        
        # 计算平均准确性
        avg_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0.0
        
        passed = avg_accuracy >= 0.7
        
        suggestions = []
        if validation_failures:
            suggestions.append(f"发现 {len(validation_failures)} 个准确性存疑的知识项")
            suggestions.append("建议通过多个独立来源进行交叉验证")
            if self.accuracy_rules['expert_validation_required']:
                suggestions.append("建议邀请领域专家进行验证")
        
        return ValidationResult(
            test_name="知识准确性验证",
            passed=passed,
            score=avg_accuracy,
            details={
                'total_items': len(knowledge_items),
                'accuracy_scores': accuracy_scores,
                'average_accuracy': avg_accuracy,
                'validation_failures': validation_failures,
                'failed_items_count': len(validation_failures)
            },
            suggestions=suggestions
        )
    
    def _validate_timeliness(self, knowledge_items: List[Dict[str, Any]]) -> ValidationResult:
        """验证知识时效性"""
        logger.info("执行知识时效性评估")
        
        current_time = datetime.now()
        outdated_items = []
        freshness_scores = []
        
        for item in knowledge_items:
            timestamp_str = item.get('timestamp', '')
            if timestamp_str:
                try:
                    if isinstance(timestamp_str, str):
                        # 修复时区问题
                        if timestamp_str.endswith('Z'):
                            item_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            item_time = datetime.fromisoformat(timestamp_str)
                    else:
                        item_time = timestamp_str
                    
                    # 确保时区一致性
                    if item_time.tzinfo is not None:
                        current_time = current_time.replace(tzinfo=item_time.tzinfo)
                    
                    age_days = (current_time - item_time).days
                    
                    # 计算新鲜度分数
                    if age_days <= self.timeliness_rules['max_age_days']:
                        freshness_score = max(0, 1 - (age_days / self.timeliness_rules['max_age_days']))
                    else:
                        freshness_score = 0
                    
                    freshness_scores.append(freshness_score)
                    
                    if age_days > self.timeliness_rules['max_age_days']:
                        outdated_items.append({
                            'item_id': item.get('id'),
                            'age_days': age_days,
                            'freshness_score': freshness_score
                        })
                
                except Exception as e:
                    logger.warning(f"解析时间戳失败: {timestamp_str}, 错误: {e}")
                    freshness_scores.append(0)
            else:
                freshness_scores.append(0)
                outdated_items.append({
                    'item_id': item.get('id'),
                    'age_days': float('inf'),
                    'freshness_score': 0,
                    'issue': 'missing_timestamp'
                })
        
        # 计算时效性分数
        avg_freshness = statistics.mean(freshness_scores) if freshness_scores else 0.0
        
        passed = avg_freshness >= 0.6
        
        suggestions = []
        if outdated_items:
            suggestions.append(f"发现 {len(outdated_items)} 个过时知识项")
            suggestions.append("建议更新或删除过时的知识信息")
            suggestions.append("建立定期的知识更新机制")
        
        return ValidationResult(
            test_name="知识时效性评估",
            passed=passed,
            score=avg_freshness,
            details={
                'total_items': len(knowledge_items),
                'freshness_scores': freshness_scores,
                'average_freshness': avg_freshness,
                'outdated_items': outdated_items,
                'outdated_count': len(outdated_items)
            },
            suggestions=suggestions
        )
    
    def _detect_conflicts(self, knowledge_items: List[Dict[str, Any]]) -> ValidationResult:
        """检测知识冲突"""
        logger.info("执行知识冲突检测")
        
        conflicts = []
        conflict_types = defaultdict(int)
        
        # 按主题分组
        topic_groups = defaultdict(list)
        for item in knowledge_items:
            topic = item.get('category', item.get('topic', 'unknown'))
            topic_groups[topic].append(item)
        
        # 检测每个主题内的冲突
        for topic, items in topic_groups.items():
            if len(items) < 2:
                continue
            
            # 检测数值冲突
            numeric_conflicts = self._detect_numeric_conflicts(items)
            conflicts.extend(numeric_conflicts)
            conflict_types['numeric'] += len(numeric_conflicts)
            
            # 检测语义冲突
            semantic_conflicts = self._detect_semantic_conflicts(items)
            conflicts.extend(semantic_conflicts)
            conflict_types['semantic'] += len(semantic_conflicts)
            
            # 检测时间冲突
            temporal_conflicts = self._detect_temporal_conflicts(items)
            conflicts.extend(temporal_conflicts)
            conflict_types['temporal'] += len(temporal_conflicts)
        
        # 计算冲突分数
        total_possible_conflicts = sum(len(items) * (len(items) - 1) // 2 
                                     for items in topic_groups.values())
        conflict_score = 1 - (len(conflicts) / max(total_possible_conflicts, 1))
        
        passed = len(conflicts) == 0
        
        suggestions = []
        if conflicts:
            suggestions.append(f"发现 {len(conflicts)} 个知识冲突")
            for conflict_type, count in conflict_types.items():
                if count > 0:
                    suggestions.append(f"{conflict_type}类型冲突: {count} 个")
            suggestions.append("建议分析冲突原因并确定正确信息")
            suggestions.append("建立冲突解决机制")
        
        return ValidationResult(
            test_name="知识冲突检测",
            passed=passed,
            score=conflict_score,
            details={
                'total_conflicts': len(conflicts),
                'conflict_types': dict(conflict_types),
                'conflict_details': conflicts,
                'topic_groups_analyzed': len(topic_groups)
            },
            suggestions=suggestions
        )
    
    def _calculate_quality_score(self, knowledge_items: List[Dict[str, Any]], 
                               validation_results: List[ValidationResult]) -> ValidationResult:
        """计算知识质量分数"""
        logger.info("执行知识质量评分")
        
        # 提取各项验证分数
        consistency_score = next((r.score for r in validation_results if r.test_name == "知识一致性检验"), 0)
        completeness_score = next((r.score for r in validation_results if r.test_name == "知识完整性检查"), 0)
        accuracy_score = next((r.score for r in validation_results if r.test_name == "知识准确性验证"), 0)
        timeliness_score = next((r.score for r in validation_results if r.test_name == "知识时效性评估"), 0)
        conflict_score = next((r.score for r in validation_results if r.test_name == "知识冲突检测"), 0)
        
        # 加权计算总体质量分数
        weights = {
            'consistency': 0.25,
            'completeness': 0.20,
            'accuracy': 0.25,
            'timeliness': 0.15,
            'conflict_resolution': 0.15
        }
        
        overall_quality = (
            consistency_score * weights['consistency'] +
            completeness_score * weights['completeness'] +
            accuracy_score * weights['accuracy'] +
            timeliness_score * weights['timeliness'] +
            conflict_score * weights['conflict_resolution']
        )
        
        # 质量等级评估
        quality_level = self._determine_quality_level(overall_quality)
        
        passed = overall_quality >= 0.7
        
        suggestions = []
        if overall_quality < 0.8:
            suggestions.append("知识质量需要改进，建议优先解决得分最低的方面")
        
        # 基于各维度分数提供具体建议
        scores = {
            '一致性': consistency_score,
            '完整性': completeness_score,
            '准确性': accuracy_score,
            '时效性': timeliness_score,
            '冲突解决': conflict_score
        }
        
        lowest_score = min(scores.items(), key=lambda x: x[1])
        if lowest_score[1] < 0.6:
            suggestions.append(f"重点改进{lowest_score[0]}方面，当前得分: {lowest_score[1]:.2f}")
        
        return ValidationResult(
            test_name="知识质量评分",
            passed=passed,
            score=overall_quality,
            details={
                'overall_quality': overall_quality,
                'quality_level': quality_level.value,
                'dimension_scores': scores,
                'weights': weights,
                'total_items': len(knowledge_items)
            },
            suggestions=suggestions
        )
    
    def _generate_repair_suggestions(self, validation_results: List[ValidationResult]) -> List[str]:
        """生成修复建议"""
        suggestions = []
        
        # 收集所有建议
        all_suggestions = []
        for result in validation_results:
            all_suggestions.extend(result.suggestions)
        
        # 去重并排序
        unique_suggestions = list(set(all_suggestions))
        
        # 按优先级排序
        priority_keywords = {
            'critical': ['冲突', '矛盾', '严重', '错误'],
            'high': ['过时', '缺失', '不完整', '准确性'],
            'medium': ['连贯性', '一致性', '优化'],
            'low': ['改进', '建议', '可以']
        }
        
        def get_priority(suggestion):
            for level, keywords in priority_keywords.items():
                if any(keyword in suggestion for keyword in keywords):
                    return {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[level]
            return 4
        
        sorted_suggestions = sorted(unique_suggestions, key=get_priority)
        
        return sorted_suggestions
    
    def _calculate_overall_score(self, validation_results: List[ValidationResult]) -> float:
        """计算总体质量分数"""
        if not validation_results:
            return 0.0
        
        # 使用质量评分的结果作为总体分数
        quality_result = next((r for r in validation_results if r.test_name == "知识质量评分"), None)
        if quality_result:
            return quality_result.score
        
        # 如果没有质量评分结果，计算平均值
        return statistics.mean([r.score for r in validation_results])
    
    def _determine_quality_level(self, score: float) -> QualityScore:
        """确定质量等级"""
        if score >= 90:
            return QualityScore.EXCELLENT
        elif score >= 80:
            return QualityScore.GOOD
        elif score >= 70:
            return QualityScore.FAIR
        elif score >= 60:
            return QualityScore.POOR
        else:
            return QualityScore.CRITICAL
    
    def _generate_summary(self, validation_results: List[ValidationResult], 
                         knowledge_items: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成验证摘要"""
        passed_tests = sum(1 for r in validation_results if r.passed)
        total_tests = len(validation_results)
        
        summary = {
            '验证概览': {
                '总测试数量': total_tests,
                '通过测试数量': passed_tests,
                '通过率': f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
            },
            '知识库统计': {
                '知识项总数': len(knowledge_items) if knowledge_items else 0,
                '验证时间': datetime.now().isoformat(),
                '验证级别': self.validation_level.value
            },
            '关键指标': {}
        }
        
        # 添加关键指标
        for result in validation_results:
            summary['关键指标'][result.test_name] = {
                '得分': f"{result.score:.2f}",
                '状态': '通过' if result.passed else '失败'
            }
        
        return summary
    
    # 辅助方法
    def _check_logical_contradiction(self, item1: Dict, item2: Dict) -> bool:
        """检查逻辑矛盾"""
        # 简化的矛盾检测逻辑
        content1 = str(item1.get('content', '')).lower()
        content2 = str(item2.get('content', '')).lower()
        
        # 检测否定词矛盾
        negation_words = ['不', '不是', '没有', '非', '否']
        for word in negation_words:
            if word in content1 and word in content2:
                return True
        
        return False
    
    def _calculate_coherence(self, item1: Dict, item2: Dict) -> float:
        """计算连贯性分数"""
        # 简化的连贯性计算
        content1 = str(item1.get('content', ''))
        content2 = str(item2.get('content', ''))
        
        # 计算词汇重叠度
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return overlap / union if union > 0 else 0.0
    
    def _evaluate_source_reliability(self, source: str) -> float:
        """评估来源可靠性"""
        if not source:
            return 0.5
        
        # 简化的来源可靠性评估
        reliable_sources = ['权威机构', '学术期刊', '官方发布', '专家认证']
        moderate_sources = ['新闻报道', '行业报告', '数据分析']
        
        source_lower = source.lower()
        
        for reliable in reliable_sources:
            if reliable in source:
                return 0.9
        
        for moderate in moderate_sources:
            if moderate in source:
                return 0.7
        
        return 0.5
    
    def _perform_cross_validation(self, item: Dict, all_items: List[Dict]) -> float:
        """执行交叉验证"""
        # 简化的交叉验证
        content = str(item.get('content', ''))
        similar_items = 0
        
        for other_item in all_items:
            if other_item == item:
                continue
            
            other_content = str(other_item.get('content', ''))
            
            # 计算相似度
            if self._calculate_similarity(content, other_content) > 0.7:
                similar_items += 1
        
        # 相似项比例作为验证分数
        validation_score = min(1.0, similar_items / max(len(all_items) - 1, 1))
        return validation_score
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_numeric_conflicts(self, items: List[Dict]) -> List[Dict]:
        """检测数值冲突"""
        conflicts = []
        
        # 提取数值信息
        numeric_data = []
        for item in items:
            content = str(item.get('content', ''))
            numbers = re.findall(r'\d+\.?\d*', content)
            if numbers:
                numeric_data.append({
                    'item_id': item.get('id'),
                    'numbers': [float(n) for n in numbers]
                })
        
        # 检测数值冲突
        for i, data1 in enumerate(numeric_data):
            for data2 in numeric_data[i+1:]:
                for num1 in data1['numbers']:
                    for num2 in data2['numbers']:
                        if abs(num1 - num2) / max(abs(num1), abs(num2), 1) > 0.1:  # 10%差异
                            conflicts.append({
                                'type': 'numeric',
                                'item1_id': data1['item_id'],
                                'item2_id': data2['item_id'],
                                'value1': num1,
                                'value2': num2,
                                'difference': abs(num1 - num2)
                            })
        
        return conflicts
    
    def _detect_semantic_conflicts(self, items: List[Dict]) -> List[Dict]:
        """检测语义冲突"""
        conflicts = []
        
        # 简化的语义冲突检测
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                content1 = str(item1.get('content', '')).lower()
                content2 = str(item2.get('content', '')).lower()
                
                # 检测相反的表述
                opposite_pairs = [
                    ('增加', '减少'),
                    ('上升', '下降'),
                    ('正面', '负面'),
                    ('支持', '反对'),
                    ('是', '否')
                ]
                
                for word1, word2 in opposite_pairs:
                    if (word1 in content1 and word2 in content2) or \
                       (word2 in content1 and word1 in content2):
                        conflicts.append({
                            'type': 'semantic',
                            'item1_id': item1.get('id'),
                            'item2_id': item2.get('id'),
                            'conflict_words': (word1, word2)
                        })
        
        return conflicts
    
    def _detect_temporal_conflicts(self, items: List[Dict]) -> List[Dict]:
        """检测时间冲突"""
        conflicts = []
        
        timestamps = []
        for item in items:
            timestamp_str = item.get('timestamp', '')
            if timestamp_str:
                try:
                    if isinstance(timestamp_str, str):
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = timestamp_str
                    timestamps.append({
                        'item_id': item.get('id'),
                        'timestamp': timestamp
                    })
                except:
                    continue
        
        # 检测时间逻辑冲突
        for i, ts1 in enumerate(timestamps):
            for ts2 in timestamps[i+1:]:
                time_diff = abs((ts1['timestamp'] - ts2['timestamp']).days)
                if time_diff < 1:  # 同一事件的不同时间表述
                    conflicts.append({
                        'type': 'temporal',
                        'item1_id': ts1['item_id'],
                        'item2_id': ts2['item_id'],
                        'time_difference': time_diff
                    })
        
        return conflicts
    
    def visualize_validation_results(self, report: KnowledgeQualityReport, 
                                   output_path: str = None) -> str:
        """
        可视化验证结果
        
        Args:
            report: 知识质量报告
            output_path: 输出路径
            
        Returns:
            可视化文件路径
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"knowledge_validation_report_{timestamp}.png"
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('知识验证结果可视化报告', fontsize=16, fontweight='bold')
        
        # 1. 总体质量分数雷达图
        ax1 = axes[0, 0]
        self._plot_quality_radar(report, ax1)
        
        # 2. 各维度得分柱状图
        ax2 = axes[0, 1]
        self._plot_dimension_scores(report, ax2)
        
        # 3. 验证状态饼图
        ax3 = axes[1, 0]
        self._plot_validation_status(report, ax3)
        
        # 4. 质量趋势图
        ax4 = axes[1, 1]
        self._plot_quality_trend(ax4)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"验证结果可视化已保存到: {output_path}")
        return output_path
    
    def _plot_quality_radar(self, report: KnowledgeQualityReport, ax):
        """绘制质量雷达图"""
        # 提取维度分数
        dimension_scores = report.summary.get('关键指标', {})
        
        if not dimension_scores:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('质量雷达图')
            return
        
        categories = list(dimension_scores.keys())
        scores = [float(dimension_scores[cat]['得分']) for cat in categories]
        
        # 计算角度
        angles = [n / float(len(categories)) * 2 * math.pi for n in range(len(categories))]
        angles += angles[:1]  # 闭合
        scores += scores[:1]  # 闭合
        
        # 绘制雷达图
        ax.plot(angles, scores, 'o-', linewidth=2, label='当前分数')
        ax.fill(angles, scores, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('质量雷达图')
        ax.grid(True)
    
    def _plot_dimension_scores(self, report: KnowledgeQualityReport, ax):
        """绘制维度得分图"""
        dimension_scores = report.summary.get('关键指标', {})
        
        if not dimension_scores:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('各维度得分')
            return
        
        categories = list(dimension_scores.keys())
        scores = [float(dimension_scores[cat]['得分']) for cat in categories]
        colors = ['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in scores]
        
        bars = ax.bar(categories, scores, color=colors, alpha=0.7)
        ax.set_title('各维度得分')
        ax.set_ylabel('得分')
        ax.set_ylim(0, 100)
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{score:.1f}', ha='center', va='bottom')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_validation_status(self, report: KnowledgeQualityReport, ax):
        """绘制验证状态图"""
        passed = sum(1 for result in report.validation_results if result.passed)
        failed = len(report.validation_results) - passed
        
        if passed + failed == 0:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('验证状态')
            return
        
        labels = ['通过', '失败']
        sizes = [passed, failed]
        colors = ['lightgreen', 'lightcoral']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('验证状态分布')
    
    def _plot_quality_trend(self, ax):
        """绘制质量趋势图"""
        if len(self.validation_history) < 2:
            ax.text(0.5, 0.5, '历史数据不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('质量趋势')
            return
        
        # 提取历史数据
        timestamps = [report.generated_at for report in self.validation_history]
        scores = [report.overall_score for report in self.validation_history]
        
        ax.plot(timestamps, scores, marker='o', linewidth=2, markersize=6)
        ax.set_title('质量趋势')
        ax.set_ylabel('质量分数')
        ax.set_xlabel('时间')
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)
    
    def generate_detailed_report(self, report: KnowledgeQualityReport, 
                               output_path: str = None) -> str:
        """
        生成详细的验证报告
        
        Args:
            report: 知识质量报告
            output_path: 输出路径
            
        Returns:
            报告文件路径
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"knowledge_validation_detailed_report_{timestamp}.json"
        
        # 构建详细报告
        detailed_report = {
            '报告信息': {
                '生成时间': report.generated_at.isoformat(),
                '总体质量分数': report.overall_score,
                '质量等级': report.quality_level.value,
                '验证级别': self.validation_level.value
            },
            '执行摘要': {
                '验证结果概览': report.summary,
                '关键建议': report.recommendations[:5]  # 前5条建议
            },
            '详细验证结果': []
        }
        
        # 添加每个验证项的详细信息
        for result in report.validation_results:
            detailed_result = {
                '测试名称': result.test_name,
                '测试状态': '通过' if result.passed else '失败',
                '得分': result.score,
                '详细信息': result.details,
                '建议': result.suggestions,
                '执行时间': result.timestamp.isoformat()
            }
            detailed_report['详细验证结果'].append(detailed_result)
        
        # 添加修复建议
        detailed_report['修复建议'] = {
            '优先级建议': report.recommendations,
            '实施计划': self._generate_implementation_plan(report)
        }
        
        # 添加质量改进建议
        detailed_report['质量改进建议'] = self._generate_quality_improvement_suggestions(report)
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细验证报告已保存到: {output_path}")
        return output_path
    
    def _generate_implementation_plan(self, report: KnowledgeQualityReport) -> List[Dict]:
        """生成实施计划"""
        plan = []
        
        # 基于验证结果生成实施计划
        for result in report.validation_results:
            if not result.passed:
                plan.append({
                    '任务': f"修复{result.test_name}",
                    '优先级': '高' if result.score < 0.5 else '中',
                    '预估工作量': '2-5天',
                    '具体步骤': result.suggestions[:3]  # 前3条建议
                })
        
        return plan
    
    def _generate_quality_improvement_suggestions(self, report: KnowledgeQualityReport) -> Dict[str, List[str]]:
        """生成质量改进建议"""
        suggestions = {
            '短期改进': [],
            '中期改进': [],
            '长期改进': []
        }
        
        # 基于质量分数提供建议
        if report.overall_score < 60:
            suggestions['短期改进'].extend([
                '立即进行知识冲突检测和解决',
                '补充缺失的关键信息',
                '验证知识来源的可靠性'
            ])
        elif report.overall_score < 80:
            suggestions['短期改进'].extend([
                '更新过时的知识信息',
                '提高知识项的完整性'
            ])
        
        suggestions['中期改进'].extend([
            '建立知识质量监控机制',
            '实施定期的知识验证流程',
            '优化知识分类和组织结构'
        ])
        
        suggestions['长期改进'].extend([
            '建立自动化知识验证系统',
            '实施机器学习辅助的质量评估',
            '建立知识图谱和关联分析'
        ])
        
        return suggestions
    
    def optimize_knowledge_base(self, knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        优化知识库
        
        Args:
            knowledge_items: 原始知识项
            
        Returns:
            优化后的知识项
        """
        logger.info("开始优化知识库")
        
        optimized_items = []
        
        for item in knowledge_items:
            optimized_item = item.copy()
            
            # 1. 清理和标准化
            optimized_item = self._standardize_item(optimized_item)
            
            # 2. 补充缺失信息
            optimized_item = self._enrich_item(optimized_item)
            
            # 3. 优化内容质量
            optimized_item = self._optimize_content(optimized_item)
            
            optimized_items.append(optimized_item)
        
        # 4. 去重和合并相似项
        optimized_items = self._deduplicate_items(optimized_items)
        
        logger.info(f"知识库优化完成，从 {len(knowledge_items)} 项优化到 {len(optimized_items)} 项")
        
        return optimized_items
    
    def _standardize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """标准化知识项"""
        standardized = item.copy()
        
        # 标准化时间戳
        if 'timestamp' in standardized and standardized['timestamp']:
            try:
                if isinstance(standardized['timestamp'], str):
                    dt = datetime.fromisoformat(standardized['timestamp'].replace('Z', '+00:00'))
                    standardized['timestamp'] = dt.isoformat()
            except:
                standardized['timestamp'] = datetime.now().isoformat()
        
        # 标准化ID
        if 'id' not in standardized or not standardized['id']:
            content_hash = hashlib.md5(str(standardized.get('content', '')).encode()).hexdigest()[:8]
            standardized['id'] = f"item_{content_hash}"
        
        # 标准化分类
        if 'category' in standardized:
            standardized['category'] = standardized['category'].lower().strip()
        
        return standardized
    
    def _enrich_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """丰富知识项信息"""
        enriched = item.copy()
        
        # 补充置信度
        if 'confidence' not in enriched or enriched['confidence'] is None:
            enriched['confidence'] = self._estimate_confidence(enriched)
        
        # 补充关键词
        if 'keywords' not in enriched:
            enriched['keywords'] = self._extract_keywords(enriched.get('content', ''))
        
        # 补充摘要
        if 'summary' not in enriched:
            enriched['summary'] = self._generate_content_summary(enriched.get('content', ''))
        
        return enriched
    
    def _estimate_confidence(self, item: Dict[str, Any]) -> float:
        """估计置信度"""
        confidence = 0.5  # 基础置信度
        
        # 基于来源调整
        source = item.get('source', '')
        if source:
            if any(keyword in source.lower() for keyword in ['权威', '官方', '学术']):
                confidence += 0.3
            elif any(keyword in source.lower() for keyword in ['新闻', '报告']):
                confidence += 0.1
        
        # 基于内容完整性调整
        content = item.get('content', '')
        if len(content) > 100:
            confidence += 0.1
        if len(content) > 500:
            confidence += 0.1
        
        # 基于元数据调整
        if item.get('category'):
            confidence += 0.05
        if item.get('timestamp'):
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _extract_keywords(self, content: str) -> List[str]:
        """提取关键词"""
        # 简化的关键词提取
        words = content.split()
        
        # 过滤停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]
        
        # 返回前10个关键词
        return keywords[:10]
    
    def _generate_content_summary(self, content: str) -> str:
        """生成摘要"""
        if len(content) <= 100:
            return content
        
        # 简化的摘要生成：取前100个字符
        return content[:100] + "..." if len(content) > 100 else content
    
    def _optimize_content(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """优化内容质量"""
        optimized = item.copy()
        content = optimized.get('content', '')
        
        # 清理内容
        content = content.strip()
        
        # 移除多余的空格
        content = re.sub(r'\s+', ' ', content)
        
        # 确保内容以句号结尾
        if content and not content.endswith(('.', '。', '!', '！', '?', '？')):
            content += '。'
        
        optimized['content'] = content
        
        return optimized
    
    def _deduplicate_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重和合并相似项"""
        unique_items = []
        seen_content_hashes = set()
        
        for item in items:
            # 计算内容哈希
            content = item.get('content', '')
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_content_hashes:
                seen_content_hashes.add(content_hash)
                unique_items.append(item)
            else:
                # 找到相似项并合并信息
                for existing_item in unique_items:
                    existing_hash = hashlib.md5(existing_item.get('content', '').encode()).hexdigest()
                    if existing_hash == content_hash:
                        # 合并元数据
                        for key, value in item.items():
                            if key not in existing_item or not existing_item[key]:
                                existing_item[key] = value
                        break
        
        return unique_items


def create_sample_knowledge_data() -> List[Dict[str, Any]]:
    """创建示例知识数据"""
    return [
        {
            'id': 'knowledge_001',
            'content': '人工智能是计算机科学的一个分支，它试图理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。',
            'timestamp': '2024-01-15T10:30:00Z',
            'source': '学术期刊',
            'category': 'technology',
            'confidence': 0.9,
            'keywords': ['人工智能', '计算机科学', '智能机器']
        },
        {
            'id': 'knowledge_002',
            'content': '机器学习是人工智能的一个子集，它使计算机有能力在不被明确编程的情况下进行学习。',
            'timestamp': '2024-02-20T14:15:00Z',
            'source': '技术报告',
            'category': 'technology',
            'confidence': 0.85,
            'keywords': ['机器学习', '人工智能', '学习算法']
        },
        {
            'id': 'knowledge_003',
            'content': '深度学习是机器学习的一个子集，它基于人工神经网络的表征学习方法。',
            'timestamp': '2024-03-10T09:45:00Z',
            'source': '学术期刊',
            'category': 'technology',
            'confidence': 0.88,
            'keywords': ['深度学习', '神经网络', '机器学习']
        },
        {
            'id': 'knowledge_004',
            'content': '自然语言处理是人工智能的一个分支，专注于计算机与人类语言之间的交互。',
            'timestamp': '2024-04-05T16:20:00Z',
            'source': '权威机构',
            'category': 'technology',
            'confidence': 0.92,
            'keywords': ['自然语言处理', 'NLP', '人机交互']
        },
        {
            'id': 'knowledge_005',
            'content': '计算机视觉是使计算机能够识别和理解视觉世界的方法和技术。',
            'timestamp': '2024-05-12T11:10:00Z',
            'source': '行业报告',
            'category': 'technology',
            'confidence': 0.87,
            'keywords': ['计算机视觉', '图像识别', '视觉处理']
        }
    ]


def main():
    """主函数 - 演示知识验证器功能"""
    print("=== C8知识验证器演示 ===\n")
    
    # 创建知识验证器
    validator = KnowledgeValidator(ValidationLevel.COMPREHENSIVE)
    
    # 创建示例数据
    knowledge_data = create_sample_knowledge_data()
    print(f"创建了 {len(knowledge_data)} 个示例知识项")
    
    # 执行知识验证
    print("\n开始知识验证...")
    report = validator.validate_knowledge_base(knowledge_data)
    
    # 打印验证结果摘要
    print(f"\n=== 验证结果摘要 ===")
    print(f"总体质量分数: {report.overall_score:.2f}")
    print(f"质量等级: {report.quality_level.value}")
    print(f"验证通过率: {sum(1 for r in report.validation_results if r.passed)}/{len(report.validation_results)}")
    
    # 打印各项验证结果
    print(f"\n=== 详细验证结果 ===")
    for result in report.validation_results:
        status = "✓ 通过" if result.passed else "✗ 失败"
        print(f"{result.test_name}: {status} (得分: {result.score:.2f})")
        if result.suggestions:
            print(f"  建议: {result.suggestions[0]}")
    
    # 生成可视化报告
    print(f"\n生成可视化报告...")
    viz_path = validator.visualize_validation_results(report)
    print(f"可视化报告已保存: {viz_path}")
    
    # 生成详细报告
    print(f"\n生成详细报告...")
    detailed_path = validator.generate_detailed_report(report)
    print(f"详细报告已保存: {detailed_path}")
    
    # 优化知识库
    print(f"\n优化知识库...")
    optimized_data = validator.optimize_knowledge_base(knowledge_data)
    print(f"优化完成，从 {len(knowledge_data)} 项优化到 {len(optimized_data)} 项")
    
    print(f"\n=== 知识验证器演示完成 ===")


if __name__ == "__main__":
    main()