#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C9 知识状态聚合器
================

智能知识融合与状态管理系统，实现多源知识融合、状态评估、可信度计算等功能。

主要功能：
1. 多源知识融合 - 整合来自不同来源的知识信息
2. 知识状态评估 - 量化评估知识状态质量
3. 知识可信度计算 - 计算知识来源的可信度
4. 知识优先级排序 - 基于重要性和时效性排序
5. 知识状态历史记录 - 记录知识状态变化历史
6. 知识状态报告生成 - 生成可视化报告
7. 知识状态预警 - 监控知识状态异常


创建时间: 2025-11-05
"""

import json
import sqlite3
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeSource:
    """知识来源数据结构"""
    source_id: str
    name: str
    source_type: str  # market, news, technical, fundamental, sentiment
    reliability_score: float  # 0-1，可靠性评分
    update_frequency: int  # 更新频率（小时）
    data_quality: float  # 0-1，数据质量评分
    last_updated: datetime
    metadata: Dict[str, Any]


@dataclass
class KnowledgeItem:
    """知识项数据结构"""
    item_id: str
    content: str
    source_id: str
    category: str
    importance: float  # 0-1，重要性评分
    freshness: float  # 0-1，时效性评分
    confidence: float  # 0-1，置信度
    timestamp: datetime
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class KnowledgeState:
    """知识状态数据结构"""
    state_id: str
    timestamp: datetime
    overall_quality: float  # 0-1，总体质量
    coverage_score: float  # 0-1，覆盖度评分
    coherence_score: float  # 0-1，一致性评分
    freshness_score: float  # 0-1，时效性评分
    reliability_score: float  # 0-1，可靠性评分
    knowledge_items: List[KnowledgeItem]
    source_status: Dict[str, Dict[str, Any]]
    alerts: List[Dict[str, Any]]


class KnowledgeFusionEngine:
    """知识融合引擎"""
    
    def __init__(self):
        self.fusion_weights = {
            'reliability': 0.3,
            'freshness': 0.25,
            'importance': 0.2,
            'confidence': 0.15,
            'data_quality': 0.1
        }
    
    def fuse_knowledge(self, knowledge_items: List[KnowledgeItem], 
                      sources: Dict[str, KnowledgeSource]) -> Dict[str, Any]:
        """融合多源知识"""
        try:
            fused_knowledge = {}
            
            # 按类别分组
            categorized_items = defaultdict(list)
            for item in knowledge_items:
                categorized_items[item.category].append(item)
            
            # 对每个类别进行融合
            for category, items in categorized_items.items():
                fused_items = self._fuse_category_items(items, sources)
                fused_knowledge[category] = fused_items
            
            # 计算整体融合质量
            fusion_quality = self._calculate_fusion_quality(fused_knowledge, sources)
            
            return {
                'fused_knowledge': fused_knowledge,
                'fusion_quality': fusion_quality,
                'fusion_timestamp': datetime.now(),
                'items_processed': len(knowledge_items),
                'sources_used': len(sources)
            }
            
        except Exception as e:
            logger.error(f"知识融合失败: {str(e)}")
            return {'error': str(e)}
    
    def _fuse_category_items(self, items: List[KnowledgeItem], 
                           sources: Dict[str, KnowledgeSource]) -> List[Dict[str, Any]]:
        """融合同类知识项"""
        if not items:
            return []
        
        # 计算每个知识项的加权分数
        weighted_scores = []
        for item in items:
            source = sources.get(item.source_id)
            if not source:
                continue
            
            # 计算加权分数
            score = (
                item.confidence * self.fusion_weights['confidence'] +
                item.freshness * self.fusion_weights['freshness'] +
                item.importance * self.fusion_weights['importance'] +
                source.reliability_score * self.fusion_weights['reliability'] +
                source.data_quality * self.fusion_weights['data_quality']
            )
            
            weighted_scores.append({
                'item': item,
                'weighted_score': score,
                'source_weight': source.reliability_score
            })
        
        # 按分数排序并选择高质量项
        weighted_scores.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # 去重和筛选
        fused_items = []
        seen_content = set()
        
        for score_item in weighted_scores[:50]:  # 限制数量
            content_hash = hash(score_item['item'].content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                fused_items.append({
                    'content': score_item['item'].content,
                    'category': score_item['item'].category,
                    'confidence': score_item['item'].confidence,
                    'importance': score_item['item'].importance,
                    'freshness': score_item['item'].freshness,
                    'source_id': score_item['item'].source_id,
                    'weighted_score': score_item['weighted_score'],
                    'timestamp': score_item['item'].timestamp
                })
        
        return fused_items
    
    def _calculate_fusion_quality(self, fused_knowledge: Dict[str, Any], 
                                sources: Dict[str, KnowledgeSource]) -> float:
        """计算融合质量"""
        try:
            total_score = 0
            total_weight = 0
            
            for category, items in fused_knowledge.items():
                if not items:
                    continue
                
                category_score = np.mean([item['weighted_score'] for item in items])
                category_weight = len(items)
                
                total_score += category_score * category_weight
                total_weight += category_weight
            
            if total_weight == 0:
                return 0.0
            
            return min(total_score / total_weight, 1.0)
            
        except Exception as e:
            logger.error(f"计算融合质量失败: {str(e)}")
            return 0.0


class KnowledgeStateEvaluator:
    """知识状态评估器"""
    
    def __init__(self):
        self.evaluation_weights = {
            'coverage': 0.3,
            'coherence': 0.25,
            'freshness': 0.2,
            'reliability': 0.15,
            'completeness': 0.1
        }
    
    def evaluate_state(self, knowledge_items: List[KnowledgeItem], 
                      sources: Dict[str, KnowledgeSource]) -> Dict[str, float]:
        """评估知识状态"""
        try:
            evaluation = {}
            
            # 覆盖度评估
            evaluation['coverage_score'] = self._evaluate_coverage(knowledge_items)
            
            # 一致性评估
            evaluation['coherence_score'] = self._evaluate_coherence(knowledge_items)
            
            # 时效性评估
            evaluation['freshness_score'] = self._evaluate_freshness(knowledge_items)
            
            # 可靠性评估
            evaluation['reliability_score'] = self._evaluate_reliability(knowledge_items, sources)
            
            # 完整性评估
            evaluation['completeness_score'] = self._evaluate_completeness(knowledge_items)
            
            # 计算总体质量
            evaluation['overall_quality'] = self._calculate_overall_quality(evaluation)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"知识状态评估失败: {str(e)}")
            return {'error': str(e)}
    
    def _evaluate_coverage(self, knowledge_items: List[KnowledgeItem]) -> float:
        """评估知识覆盖度"""
        if not knowledge_items:
            return 0.0
        
        # 按类别统计
        categories = set(item.category for item in knowledge_items)
        total_categories = len(categories)
        
        # 计算每个类别的项目数量
        category_counts = defaultdict(int)
        for item in knowledge_items:
            category_counts[item.category] += 1
        
        # 评估覆盖均衡性
        counts = list(category_counts.values())
        if len(counts) == 0:
            return 0.0
        
        # 使用基尼系数评估均衡性
        counts.sort()
        n = len(counts)
        cumsum = np.cumsum(counts)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        # 结合类别数量和均衡性
        coverage_score = (total_categories / 10) * (1 - gini)  # 假设最多10个类别
        return min(coverage_score, 1.0)
    
    def _evaluate_coherence(self, knowledge_items: List[KnowledgeItem]) -> float:
        """评估知识一致性"""
        if len(knowledge_items) < 2:
            return 1.0
        
        # 计算内容相似度（简化版）
        coherence_scores = []
        
        for i in range(len(knowledge_items)):
            for j in range(i + 1, len(knowledge_items)):
                item1, item2 = knowledge_items[i], knowledge_items[j]
                
                # 如果是同一类别且时间接近，认为一致
                if (item1.category == item2.category and
                    abs((item1.timestamp - item2.timestamp).total_seconds()) < 3600):  # 1小时
                    coherence_scores.append(0.8)
                elif item1.category == item2.category:
                    coherence_scores.append(0.6)
                else:
                    coherence_scores.append(0.3)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _evaluate_freshness(self, knowledge_items: List[KnowledgeItem]) -> float:
        """评估知识时效性"""
        if not knowledge_items:
            return 0.0
        
        now = datetime.now()
        freshness_scores = []
        
        for item in knowledge_items:
            age_hours = (now - item.timestamp).total_seconds() / 3600
            # 使用指数衰减计算新鲜度
            freshness = np.exp(-age_hours / 24)  # 24小时半衰期
            freshness_scores.append(freshness)
        
        return np.mean(freshness_scores)
    
    def _evaluate_reliability(self, knowledge_items: List[KnowledgeItem], 
                            sources: Dict[str, KnowledgeSource]) -> float:
        """评估知识可靠性"""
        if not knowledge_items:
            return 0.0
        
        reliability_scores = []
        
        for item in knowledge_items:
            source = sources.get(item.source_id)
            if source:
                # 结合源可靠性和项目置信度
                reliability = (source.reliability_score + item.confidence) / 2
                reliability_scores.append(reliability)
        
        return np.mean(reliability_scores) if reliability_scores else 0.0
    
    def _evaluate_completeness(self, knowledge_items: List[KnowledgeItem]) -> float:
        """评估知识完整性"""
        if not knowledge_items:
            return 0.0
        
        # 检查必要字段的完整性
        complete_items = 0
        
        for item in knowledge_items:
            if (item.content and 
                item.category and 
                len(item.content.strip()) > 10 and
                len(item.tags) > 0):
                complete_items += 1
        
        return complete_items / len(knowledge_items)
    
    def _calculate_overall_quality(self, evaluation: Dict[str, float]) -> float:
        """计算总体质量分数"""
        if 'error' in evaluation:
            return 0.0
        
        total_score = 0
        total_weight = 0
        
        for metric, score in evaluation.items():
            if metric != 'overall_quality' and metric != 'error':
                weight = self.evaluation_weights.get(metric.replace('_score', ''), 0.1)
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


class KnowledgeCredibilityCalculator:
    """知识可信度计算器"""
    
    def __init__(self):
        self.credibility_factors = {
            'source_reliability': 0.4,
            'data_consistency': 0.25,
            'historical_accuracy': 0.2,
            'expert_validation': 0.1,
            'cross_reference': 0.05
        }
    
    def calculate_credibility(self, knowledge_items: List[KnowledgeItem], 
                            sources: Dict[str, KnowledgeSource],
                            historical_data: Optional[Dict] = None) -> Dict[str, float]:
        """计算知识可信度"""
        try:
            credibility_scores = {}
            
            for item in knowledge_items:
                credibility_score = self._calculate_item_credibility(
                    item, sources, historical_data
                )
                credibility_scores[item.item_id] = credibility_score
            
            # 计算平均可信度
            overall_credibility = np.mean(list(credibility_scores.values())) if credibility_scores else 0.0
            
            return {
                'individual_scores': credibility_scores,
                'overall_credibility': overall_credibility,
                'credibility_distribution': self._get_credibility_distribution(credibility_scores)
            }
            
        except Exception as e:
            logger.error(f"可信度计算失败: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_item_credibility(self, item: KnowledgeItem, 
                                  sources: Dict[str, KnowledgeSource],
                                  historical_data: Optional[Dict] = None) -> float:
        """计算单个知识项的可信度"""
        credibility_score = 0.0
        
        # 1. 来源可靠性 (40%)
        source = sources.get(item.source_id)
        if source:
            credibility_score += source.reliability_score * self.credibility_factors['source_reliability']
        
        # 2. 数据一致性 (25%)
        consistency_score = self._calculate_consistency(item, sources)
        credibility_score += consistency_score * self.credibility_factors['data_consistency']
        
        # 3. 历史准确性 (20%)
        historical_score = self._calculate_historical_accuracy(item, historical_data)
        credibility_score += historical_score * self.credibility_factors['historical_accuracy']
        
        # 4. 专家验证 (10%)
        expert_score = item.metadata.get('expert_validated', 0.5)
        credibility_score += expert_score * self.credibility_factors['expert_validation']
        
        # 5. 交叉引用 (5%)
        cross_ref_score = len(item.metadata.get('cross_references', [])) / 10  # 假设最多10个引用
        credibility_score += min(cross_ref_score, 1.0) * self.credibility_factors['cross_reference']
        
        return min(credibility_score, 1.0)
    
    def _calculate_consistency(self, item: KnowledgeItem, sources: Dict[str, KnowledgeSource]) -> float:
        """计算数据一致性"""
        # 简化实现：检查同类别项目的置信度一致性
        source = sources.get(item.source_id)
        if not source:
            return 0.5
        
        # 基于源数据质量评估一致性
        return source.data_quality
    
    def _calculate_historical_accuracy(self, item: KnowledgeItem, 
                                     historical_data: Optional[Dict] = None) -> float:
        """计算历史准确性"""
        if not historical_data:
            return 0.7  # 默认中等准确性
        
        # 简化实现：基于历史表现
        source_history = historical_data.get(item.source_id, {})
        accuracy_rate = source_history.get('accuracy_rate', 0.7)
        
        return accuracy_rate
    
    def _get_credibility_distribution(self, scores: Dict[str, float]) -> Dict[str, float]:
        """获取可信度分布"""
        if not scores:
            return {'high': 0.0, 'medium': 0.0, 'low': 0.0}
        
        values = list(scores.values())
        
        return {
            'high': len([s for s in values if s >= 0.8]) / len(values),
            'medium': len([s for s in values if 0.5 <= s < 0.8]) / len(values),
            'low': len([s for s in values if s < 0.5]) / len(values)
        }


class KnowledgePriorityRanker:
    """知识优先级排序器"""
    
    def __init__(self):
        self.priority_weights = {
            'importance': 0.35,
            'urgency': 0.25,
            'impact': 0.2,
            'freshness': 0.15,
            'credibility': 0.05
        }
    
    def rank_knowledge(self, knowledge_items: List[KnowledgeItem], 
                      credibility_scores: Dict[str, float],
                      context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """对知识进行优先级排序"""
        try:
            ranked_items = []
            
            for item in knowledge_items:
                priority_score = self._calculate_priority_score(
                    item, credibility_scores, context
                )
                
                ranked_items.append({
                    'item': item,
                    'priority_score': priority_score,
                    'priority_level': self._get_priority_level(priority_score)
                })
            
            # 按优先级分数排序
            ranked_items.sort(key=lambda x: x['priority_score'], reverse=True)
            
            return ranked_items
            
        except Exception as e:
            logger.error(f"知识优先级排序失败: {str(e)}")
            return []
    
    def _calculate_priority_score(self, item: KnowledgeItem, 
                                credibility_scores: Dict[str, float],
                                context: Optional[Dict[str, Any]] = None) -> float:
        """计算优先级分数"""
        priority_score = 0.0
        
        # 1. 重要性 (35%)
        priority_score += item.importance * self.priority_weights['importance']
        
        # 2. 紧急性 (25%)
        urgency_score = self._calculate_urgency(item, context)
        priority_score += urgency_score * self.priority_weights['urgency']
        
        # 3. 影响度 (20%)
        impact_score = item.metadata.get('impact_score', 0.5)
        priority_score += impact_score * self.priority_weights['impact']
        
        # 4. 时效性 (15%)
        priority_score += item.freshness * self.priority_weights['freshness']
        
        # 5. 可信度 (5%)
        credibility = credibility_scores.get(item.item_id, 0.5)
        priority_score += credibility * self.priority_weights['credibility']
        
        return priority_score
    
    def _calculate_urgency(self, item: KnowledgeItem, context: Optional[Dict[str, Any]] = None) -> float:
        """计算紧急性分数"""
        # 基于时间和类型的紧急性
        now = datetime.now()
        age_hours = (now - item.timestamp).total_seconds() / 3600
        
        # 时间紧急性（越新越紧急）
        time_urgency = np.exp(-age_hours / 12)  # 12小时半衰期
        
        # 类型紧急性
        type_urgency_map = {
            'breaking_news': 1.0,
            'market_alert': 0.9,
            'technical_signal': 0.7,
            'fundamental_data': 0.6,
            'sentiment': 0.5,
            'general': 0.3
        }
        type_urgency = type_urgency_map.get(item.category, 0.5)
        
        return (time_urgency + type_urgency) / 2
    
    def _get_priority_level(self, score: float) -> str:
        """获取优先级等级"""
        if score >= 0.8:
            return 'critical'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'


class KnowledgeStateHistory:
    """知识状态历史记录器"""
    
    def __init__(self, db_path: str = "knowledge_state_history.db"):
        self.db_path = db_path
        self.init_database()
        self.lock = threading.Lock()
    
    def init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建知识状态历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_state_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    overall_quality REAL,
                    coverage_score REAL,
                    coherence_score REAL,
                    freshness_score REAL,
                    reliability_score REAL,
                    knowledge_count INTEGER,
                    source_count INTEGER,
                    alert_count INTEGER,
                    state_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建知识项历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_item_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    state_id TEXT NOT NULL,
                    content TEXT,
                    category TEXT,
                    importance REAL,
                    confidence REAL,
                    source_id TEXT,
                    timestamp DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {str(e)}")
    
    def record_state(self, knowledge_state: KnowledgeState):
        """记录知识状态"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 记录状态
                cursor.execute('''
                    INSERT INTO knowledge_state_history 
                    (state_id, timestamp, overall_quality, coverage_score, 
                     coherence_score, freshness_score, reliability_score,
                     knowledge_count, source_count, alert_count, state_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    knowledge_state.state_id,
                    knowledge_state.timestamp,
                    knowledge_state.overall_quality,
                    knowledge_state.coverage_score,
                    knowledge_state.coherence_score,
                    knowledge_state.freshness_score,
                    knowledge_state.reliability_score,
                    len(knowledge_state.knowledge_items),
                    len(knowledge_state.source_status),
                    len(knowledge_state.alerts),
                    json.dumps(asdict(knowledge_state), default=str, ensure_ascii=False)
                ))
                
                # 记录知识项
                for item in knowledge_state.knowledge_items:
                    cursor.execute('''
                        INSERT INTO knowledge_item_history
                        (item_id, state_id, content, category, importance, 
                         confidence, source_id, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        item.item_id,
                        knowledge_state.state_id,
                        item.content,
                        item.category,
                        item.importance,
                        item.confidence,
                        item.source_id,
                        item.timestamp
                    ))
                
                conn.commit()
                conn.close()
                
                logger.info(f"知识状态已记录: {knowledge_state.state_id}")
                
        except Exception as e:
            logger.error(f"记录知识状态失败: {str(e)}")
    
    def get_state_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取历史状态"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since_date = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                SELECT * FROM knowledge_state_history 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            ''', (since_date,))
            
            columns = [description[0] for description in cursor.description]
            history = []
            
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                if record['state_data']:
                    record['state_data'] = json.loads(record['state_data'])
                history.append(record)
            
            conn.close()
            return history
            
        except Exception as e:
            logger.error(f"获取历史状态失败: {str(e)}")
            return []
    
    def get_state_trend(self, metric: str, days: int = 30) -> List[Tuple[datetime, float]]:
        """获取状态趋势"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since_date = datetime.now() - timedelta(days=days)
            
            cursor.execute(f'''
                SELECT timestamp, {metric} FROM knowledge_state_history 
                WHERE timestamp >= ? AND {metric} IS NOT NULL
                ORDER BY timestamp ASC
            ''', (since_date,))
            
            trend = [(datetime.fromisoformat(row[0]), row[1]) for row in cursor.fetchall()]
            
            conn.close()
            return trend
            
        except Exception as e:
            logger.error(f"获取状态趋势失败: {str(e)}")
            return []


class KnowledgeStateReporter:
    """知识状态报告生成器"""
    
    def __init__(self):
        self.report_templates = {
            'summary': self._generate_summary_report,
            'detailed': self._generate_detailed_report,
            'trend': self._generate_trend_report,
            'alert': self._generate_alert_report
        }
    
    def generate_report(self, knowledge_state: KnowledgeState, 
                       report_type: str = 'summary',
                       output_format: str = 'json') -> Dict[str, Any]:
        """生成知识状态报告"""
        try:
            if report_type not in self.report_templates:
                raise ValueError(f"不支持的报告类型: {report_type}")
            
            report_data = self.report_templates[report_type](knowledge_state)
            
            if output_format == 'json':
                return report_data
            elif output_format == 'html':
                return self._generate_html_report(report_data)
            else:
                return report_data
                
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
            return {'error': str(e)}
    
    def _generate_summary_report(self, knowledge_state: KnowledgeState) -> Dict[str, Any]:
        """生成摘要报告"""
        return {
            'report_type': 'summary',
            'timestamp': knowledge_state.timestamp.isoformat(),
            'overall_quality': knowledge_state.overall_quality,
            'key_metrics': {
                'coverage_score': knowledge_state.coverage_score,
                'coherence_score': knowledge_state.coherence_score,
                'freshness_score': knowledge_state.freshness_score,
                'reliability_score': knowledge_state.reliability_score
            },
            'knowledge_summary': {
                'total_items': len(knowledge_state.knowledge_items),
                'categories': list(set(item.category for item in knowledge_state.knowledge_items)),
                'top_sources': self._get_top_sources(knowledge_state.source_status)
            },
            'alerts': knowledge_state.alerts,
            'recommendations': self._generate_recommendations(knowledge_state)
        }
    
    def _generate_detailed_report(self, knowledge_state: KnowledgeState) -> Dict[str, Any]:
        """生成详细报告"""
        # 将知识状态转换为可序列化的格式
        state_data = asdict(knowledge_state)
        # 转换datetime对象为字符串
        state_data['timestamp'] = knowledge_state.timestamp.isoformat()
        for item in state_data['knowledge_items']:
            item['timestamp'] = item['timestamp'].isoformat()
        
        return {
            'report_type': 'detailed',
            'timestamp': knowledge_state.timestamp.isoformat(),
            'state_overview': state_data,
            'knowledge_analysis': self._analyze_knowledge_items(knowledge_state.knowledge_items),
            'source_analysis': self._analyze_sources(knowledge_state.source_status),
            'quality_metrics': self._calculate_detailed_metrics(knowledge_state),
            'recommendations': self._generate_recommendations(knowledge_state)
        }
    
    def _generate_trend_report(self, knowledge_state: KnowledgeState) -> Dict[str, Any]:
        """生成趋势报告"""
        # 将知识状态转换为可序列化的格式
        state_data = asdict(knowledge_state)
        # 转换datetime对象为字符串
        state_data['timestamp'] = knowledge_state.timestamp.isoformat()
        for item in state_data['knowledge_items']:
            item['timestamp'] = item['timestamp'].isoformat()
        
        return {
            'report_type': 'trend',
            'timestamp': knowledge_state.timestamp.isoformat(),
            'current_state': state_data,
            'trend_analysis': {
                'quality_trend': 'stable',  # 简化实现
                'coverage_trend': 'improving',
                'reliability_trend': 'stable'
            },
            'predictions': self._generate_predictions(knowledge_state)
        }
    
    def _generate_alert_report(self, knowledge_state: KnowledgeState) -> Dict[str, Any]:
        """生成预警报告"""
        critical_alerts = [alert for alert in knowledge_state.alerts if alert.get('severity') == 'critical']
        
        return {
            'report_type': 'alert',
            'timestamp': knowledge_state.timestamp.isoformat(),
            'alert_summary': {
                'total_alerts': len(knowledge_state.alerts),
                'critical_alerts': len(critical_alerts),
                'alert_types': list(set(alert.get('type') for alert in knowledge_state.alerts))
            },
            'critical_alerts': critical_alerts,
            'recommended_actions': self._get_recommended_actions(knowledge_state.alerts)
        }
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """生成HTML格式报告"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>知识状态报告 - {report_data.get('report_type', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; }}
                .alert {{ color: red; font-weight: bold; }}
                .success {{ color: green; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>知识状态报告</h1>
                <p>报告类型: {report_data.get('report_type', 'Unknown')}</p>
                <p>生成时间: {report_data.get('timestamp', 'Unknown')}</p>
            </div>
            
            <h2>总体质量</h2>
            <div class="metric">
                <strong>质量分数:</strong> {report_data.get('overall_quality', 'N/A'):.2f}
            </div>
            
            <h2>关键指标</h2>
            {self._format_metrics_html(report_data.get('key_metrics', {}))}
            
            <h2>预警信息</h2>
            {self._format_alerts_html(report_data.get('alerts', []))}
        </body>
        </html>
        """
        return html_template
    
    def _get_top_sources(self, source_status: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """获取顶级数据源"""
        sources = []
        for source_id, status in source_status.items():
            sources.append({
                'source_id': source_id,
                'status': status.get('status', 'unknown'),
                'reliability': status.get('reliability', 0.0)
            })
        
        sources.sort(key=lambda x: x['reliability'], reverse=True)
        return sources[:5]  # 返回前5个
    
    def _generate_recommendations(self, knowledge_state: KnowledgeState) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if knowledge_state.overall_quality < 0.6:
            recommendations.append("建议提高数据源质量，优先使用高可靠性来源")
        
        if knowledge_state.freshness_score < 0.5:
            recommendations.append("建议增加数据更新频率，提高知识时效性")
        
        if knowledge_state.coherence_score < 0.7:
            recommendations.append("建议加强数据一致性验证，减少矛盾信息")
        
        if len(knowledge_state.alerts) > 3:
            recommendations.append("预警较多，建议检查数据源状态")
        
        return recommendations
    
    def _analyze_knowledge_items(self, items: List[KnowledgeItem]) -> Dict[str, Any]:
        """分析知识项"""
        if not items:
            return {}
        
        categories = defaultdict(int)
        importance_distribution = []
        confidence_distribution = []
        
        for item in items:
            categories[item.category] += 1
            importance_distribution.append(item.importance)
            confidence_distribution.append(item.confidence)
        
        return {
            'category_distribution': dict(categories),
            'importance_stats': {
                'mean': np.mean(importance_distribution),
                'std': np.std(importance_distribution),
                'min': np.min(importance_distribution),
                'max': np.max(importance_distribution)
            },
            'confidence_stats': {
                'mean': np.mean(confidence_distribution),
                'std': np.std(confidence_distribution),
                'min': np.min(confidence_distribution),
                'max': np.max(confidence_distribution)
            }
        }
    
    def _analyze_sources(self, source_status: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """分析数据源"""
        if not source_status:
            return {}
        
        status_counts = defaultdict(int)
        reliability_scores = []
        
        for source_id, status in source_status.items():
            status_counts[status.get('status', 'unknown')] += 1
            reliability_scores.append(status.get('reliability', 0.0))
        
        return {
            'status_distribution': dict(status_counts),
            'reliability_stats': {
                'mean': np.mean(reliability_scores),
                'std': np.std(reliability_scores),
                'min': np.min(reliability_scores),
                'max': np.max(reliability_scores)
            }
        }
    
    def _calculate_detailed_metrics(self, knowledge_state: KnowledgeState) -> Dict[str, Any]:
        """计算详细指标"""
        return {
            'quality_score': knowledge_state.overall_quality,
            'coverage_efficiency': knowledge_state.coverage_score / max(knowledge_state.overall_quality, 0.01),
            'freshness_impact': knowledge_state.freshness_score * knowledge_state.overall_quality,
            'reliability_weighted': knowledge_state.reliability_score * knowledge_state.overall_quality
        }
    
    def _generate_predictions(self, knowledge_state: KnowledgeState) -> Dict[str, Any]:
        """生成预测"""
        # 简化预测逻辑
        return {
            'quality_prediction': knowledge_state.overall_quality + np.random.normal(0, 0.1),
            'coverage_prediction': min(knowledge_state.coverage_score + 0.1, 1.0),
            'trend_direction': 'stable'
        }
    
    def _get_recommended_actions(self, alerts: List[Dict[str, Any]]) -> List[str]:
        """获取建议行动"""
        actions = []
        
        for alert in alerts:
            alert_type = alert.get('type', '')
            severity = alert.get('severity', 'medium')
            
            if severity == 'critical':
                if 'data_quality' in alert_type:
                    actions.append("立即检查数据源质量，暂停低质量数据源")
                elif 'freshness' in alert_type:
                    actions.append("紧急更新数据源，提高更新频率")
                elif 'coherence' in alert_type:
                    actions.append("立即验证数据一致性，隔离矛盾信息")
        
        return actions
    
    def _format_metrics_html(self, metrics: Dict[str, Any]) -> str:
        """格式化指标HTML"""
        html = ""
        for metric, value in metrics.items():
            html += f'<div class="metric"><strong>{metric}:</strong> {value:.2f}</div>'
        return html
    
    def _format_alerts_html(self, alerts: List[Dict[str, Any]]) -> str:
        """格式化预警HTML"""
        if not alerts:
            return '<div class="success">暂无预警信息</div>'
        
        html = ""
        for alert in alerts:
            severity = alert.get('severity', 'medium')
            alert_class = 'alert' if severity == 'critical' else ''
            html += f'<div class="{alert_class}">预警: {alert.get("message", "未知预警")}</div>'
        return html


class KnowledgeStateAlerter:
    """知识状态预警器"""
    
    def __init__(self):
        self.alert_thresholds = {
            'quality_critical': 0.3,
            'quality_warning': 0.5,
            'freshness_critical': 0.2,
            'freshness_warning': 0.4,
            'coherence_critical': 0.4,
            'coherence_warning': 0.6,
            'reliability_critical': 0.3,
            'reliability_warning': 0.5
        }
        self.alert_handlers = []
    
    def check_alerts(self, knowledge_state: KnowledgeState) -> List[Dict[str, Any]]:
        """检查预警条件"""
        alerts = []
        
        # 质量预警
        if knowledge_state.overall_quality <= self.alert_thresholds['quality_critical']:
            alerts.append({
                'type': 'quality_critical',
                'severity': 'critical',
                'message': f'知识质量严重下降: {knowledge_state.overall_quality:.2f}',
                'timestamp': datetime.now(),
                'metric': 'overall_quality',
                'value': knowledge_state.overall_quality
            })
        elif knowledge_state.overall_quality <= self.alert_thresholds['quality_warning']:
            alerts.append({
                'type': 'quality_warning',
                'severity': 'warning',
                'message': f'知识质量偏低: {knowledge_state.overall_quality:.2f}',
                'timestamp': datetime.now(),
                'metric': 'overall_quality',
                'value': knowledge_state.overall_quality
            })
        
        # 时效性预警
        if knowledge_state.freshness_score <= self.alert_thresholds['freshness_critical']:
            alerts.append({
                'type': 'freshness_critical',
                'severity': 'critical',
                'message': f'知识严重过时: {knowledge_state.freshness_score:.2f}',
                'timestamp': datetime.now(),
                'metric': 'freshness_score',
                'value': knowledge_state.freshness_score
            })
        elif knowledge_state.freshness_score <= self.alert_thresholds['freshness_warning']:
            alerts.append({
                'type': 'freshness_warning',
                'severity': 'warning',
                'message': f'知识新鲜度偏低: {knowledge_state.freshness_score:.2f}',
                'timestamp': datetime.now(),
                'metric': 'freshness_score',
                'value': knowledge_state.freshness_score
            })
        
        # 一致性预警
        if knowledge_state.coherence_score <= self.alert_thresholds['coherence_critical']:
            alerts.append({
                'type': 'coherence_critical',
                'severity': 'critical',
                'message': f'知识一致性严重问题: {knowledge_state.coherence_score:.2f}',
                'timestamp': datetime.now(),
                'metric': 'coherence_score',
                'value': knowledge_state.coherence_score
            })
        elif knowledge_state.coherence_score <= self.alert_thresholds['coherence_warning']:
            alerts.append({
                'type': 'coherence_warning',
                'severity': 'warning',
                'message': f'知识一致性偏低: {knowledge_state.coherence_score:.2f}',
                'timestamp': datetime.now(),
                'metric': 'coherence_score',
                'value': knowledge_state.coherence_score
            })
        
        # 可靠性预警
        if knowledge_state.reliability_score <= self.alert_thresholds['reliability_critical']:
            alerts.append({
                'type': 'reliability_critical',
                'severity': 'critical',
                'message': f'知识可靠性严重不足: {knowledge_state.reliability_score:.2f}',
                'timestamp': datetime.now(),
                'metric': 'reliability_score',
                'value': knowledge_state.reliability_score
            })
        elif knowledge_state.reliability_score <= self.alert_thresholds['reliability_warning']:
            alerts.append({
                'type': 'reliability_warning',
                'severity': 'warning',
                'message': f'知识可靠性偏低: {knowledge_state.reliability_score:.2f}',
                'timestamp': datetime.now(),
                'metric': 'reliability_score',
                'value': knowledge_state.reliability_score
            })
        
        # 数据源状态预警
        source_alerts = self._check_source_alerts(knowledge_state.source_status)
        alerts.extend(source_alerts)
        
        # 知识项异常预警
        item_alerts = self._check_knowledge_item_alerts(knowledge_state.knowledge_items)
        alerts.extend(item_alerts)
        
        return alerts
    
    def _check_source_alerts(self, source_status: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检查数据源预警"""
        alerts = []
        
        for source_id, status in source_status.items():
            if status.get('status') == 'offline':
                alerts.append({
                    'type': 'source_offline',
                    'severity': 'critical',
                    'message': f'数据源离线: {source_id}',
                    'timestamp': datetime.now(),
                    'source_id': source_id
                })
            elif status.get('status') == 'degraded':
                alerts.append({
                    'type': 'source_degraded',
                    'severity': 'warning',
                    'message': f'数据源性能下降: {source_id}',
                    'timestamp': datetime.now(),
                    'source_id': source_id
                })
        
        return alerts
    
    def _check_knowledge_item_alerts(self, knowledge_items: List[KnowledgeItem]) -> List[Dict[str, Any]]:
        """检查知识项预警"""
        alerts = []
        
        # 检查低置信度项目
        low_confidence_items = [item for item in knowledge_items if item.confidence < 0.3]
        if len(low_confidence_items) > len(knowledge_items) * 0.2:  # 超过20%
            alerts.append({
                'type': 'low_confidence_items',
                'severity': 'warning',
                'message': f'低置信度知识项过多: {len(low_confidence_items)}/{len(knowledge_items)}',
                'timestamp': datetime.now(),
                'count': len(low_confidence_items)
            })
        
        # 检查过时项目
        now = datetime.now()
        outdated_items = [item for item in knowledge_items 
                         if (now - item.timestamp).total_seconds() > 86400]  # 超过24小时
        if len(outdated_items) > len(knowledge_items) * 0.5:  # 超过50%
            alerts.append({
                'type': 'outdated_items',
                'severity': 'warning',
                'message': f'过时知识项过多: {len(outdated_items)}/{len(knowledge_items)}',
                'timestamp': datetime.now(),
                'count': len(outdated_items)
            })
        
        return alerts
    
    def add_alert_handler(self, handler):
        """添加预警处理器"""
        self.alert_handlers.append(handler)
    
    def process_alerts(self, alerts: List[Dict[str, Any]]):
        """处理预警"""
        for alert in alerts:
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"预警处理器执行失败: {str(e)}")


class KnowledgeStateAggregator:
    """知识状态聚合器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化知识状态聚合器"""
        self.config = config or {}
        
        # 初始化各个组件
        self.fusion_engine = KnowledgeFusionEngine()
        self.evaluator = KnowledgeStateEvaluator()
        self.credibility_calculator = KnowledgeCredibilityCalculator()
        self.priority_ranker = KnowledgePriorityRanker()
        self.history_recorder = KnowledgeStateHistory()
        self.reporter = KnowledgeStateReporter()
        self.alerter = KnowledgeStateAlerter()
        
        # 状态管理
        self.current_state: Optional[KnowledgeState] = None
        self.knowledge_sources: Dict[str, KnowledgeSource] = {}
        self.knowledge_items: List[KnowledgeItem] = []
        
        # 监控线程
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("知识状态聚合器初始化完成")
    
    def register_source(self, source: KnowledgeSource):
        """注册知识来源"""
        try:
            self.knowledge_sources[source.source_id] = source
            logger.info(f"知识来源已注册: {source.name} ({source.source_id})")
        except Exception as e:
            logger.error(f"注册知识来源失败: {str(e)}")
    
    def add_knowledge_item(self, item: KnowledgeItem):
        """添加知识项"""
        try:
            self.knowledge_items.append(item)
            logger.debug(f"知识项已添加: {item.item_id}")
        except Exception as e:
            logger.error(f"添加知识项失败: {str(e)}")
    
    def process_knowledge(self, context: Optional[Dict[str, Any]] = None) -> KnowledgeState:
        """处理知识并生成状态"""
        try:
            logger.info("开始处理知识状态...")
            
            # 1. 知识融合
            fusion_result = self.fusion_engine.fuse_knowledge(self.knowledge_items, self.knowledge_sources)
            
            # 2. 状态评估
            evaluation = self.evaluator.evaluate_state(self.knowledge_items, self.knowledge_sources)
            
            # 3. 可信度计算
            credibility_result = self.credibility_calculator.calculate_credibility(
                self.knowledge_items, self.knowledge_sources
            )
            
            # 4. 优先级排序
            ranked_knowledge = self.priority_ranker.rank_knowledge(
                self.knowledge_items, 
                credibility_result.get('individual_scores', {}),
                context
            )
            
            # 5. 预警检查
            temp_state = KnowledgeState(
                state_id=f"temp_{int(time.time())}",
                timestamp=datetime.now(),
                overall_quality=evaluation.get('overall_quality', 0.0),
                coverage_score=evaluation.get('coverage_score', 0.0),
                coherence_score=evaluation.get('coherence_score', 0.0),
                freshness_score=evaluation.get('freshness_score', 0.0),
                reliability_score=evaluation.get('reliability_score', 0.0),
                knowledge_items=self.knowledge_items,
                source_status=self._get_source_status(),
                alerts=[]
            )
            
            alerts = self.alerter.check_alerts(temp_state)
            
            # 6. 创建最终状态
            self.current_state = KnowledgeState(
                state_id=f"state_{int(time.time())}",
                timestamp=datetime.now(),
                overall_quality=evaluation.get('overall_quality', 0.0),
                coverage_score=evaluation.get('coverage_score', 0.0),
                coherence_score=evaluation.get('coherence_score', 0.0),
                freshness_score=evaluation.get('freshness_score', 0.0),
                reliability_score=evaluation.get('reliability_score', 0.0),
                knowledge_items=self.knowledge_items,
                source_status=self._get_source_status(),
                alerts=alerts
            )
            
            # 7. 记录历史
            self.history_recorder.record_state(self.current_state)
            
            # 8. 处理预警
            if alerts:
                self.alerter.process_alerts(alerts)
            
            logger.info(f"知识状态处理完成: 质量={self.current_state.overall_quality:.2f}, "
                       f"预警数={len(alerts)}")
            
            return self.current_state
            
        except Exception as e:
            logger.error(f"知识处理失败: {str(e)}")
            raise
    
    def get_current_state(self) -> Optional[KnowledgeState]:
        """获取当前知识状态"""
        return self.current_state
    
    def generate_report(self, report_type: str = 'summary', output_format: str = 'json') -> Dict[str, Any]:
        """生成知识状态报告"""
        if not self.current_state:
            raise ValueError("没有可用的知识状态，请先调用 process_knowledge()")
        
        return self.reporter.generate_report(self.current_state, report_type, output_format)
    
    def get_state_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取知识状态历史"""
        return self.history_recorder.get_state_history(days)
    
    def get_state_trend(self, metric: str, days: int = 30) -> List[Tuple[datetime, float]]:
        """获取状态趋势"""
        return self.history_recorder.get_state_trend(metric, days)
    
    def start_monitoring(self, interval: int = 300):  # 5分钟间隔
        """启动状态监控"""
        if self.monitoring_active:
            logger.warning("监控已在运行中")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"知识状态监控已启动，间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止状态监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("知识状态监控已停止")
    
    def _monitoring_loop(self, interval: int):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 自动处理知识状态
                self.process_knowledge()
                
                # 检查预警
                if self.current_state and self.current_state.alerts:
                    logger.warning(f"检测到 {len(self.current_state.alerts)} 个预警")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {str(e)}")
                time.sleep(interval)
    
    def _get_source_status(self) -> Dict[str, Dict[str, Any]]:
        """获取数据源状态"""
        status = {}
        for source_id, source in self.knowledge_sources.items():
            status[source_id] = {
                'status': 'online',  # 简化实现
                'reliability': source.reliability_score,
                'data_quality': source.data_quality,
                'last_updated': source.last_updated.isoformat(),
                'update_frequency': source.update_frequency
            }
        return status
    
    def export_state(self, file_path: str):
        """导出知识状态"""
        if not self.current_state:
            raise ValueError("没有可用的知识状态")
        
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'state': asdict(self.current_state),
                'sources': {sid: asdict(source) for sid, source in self.knowledge_sources.items()},
                'metadata': {
                    'total_items': len(self.knowledge_items),
                    'total_sources': len(self.knowledge_sources)
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"知识状态已导出到: {file_path}")
            
        except Exception as e:
            logger.error(f"导出知识状态失败: {str(e)}")
            raise
    
    def import_state(self, file_path: str):
        """导入知识状态"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 重建知识状态
            state_data = import_data['state']
            self.current_state = KnowledgeState(**state_data)
            
            # 重建数据源
            self.knowledge_sources = {}
            for sid, source_data in import_data['sources'].items():
                self.knowledge_sources[sid] = KnowledgeSource(**source_data)
            
            # 重建知识项
            self.knowledge_items = [KnowledgeItem(**item_data) 
                                  for item_data in state_data['knowledge_items']]
            
            logger.info(f"知识状态已从 {file_path} 导入")
            
        except Exception as e:
            logger.error(f"导入知识状态失败: {str(e)}")
            raise


def create_sample_data() -> Tuple[List[KnowledgeSource], List[KnowledgeItem]]:
    """创建示例数据"""
    
    # 创建示例知识来源
    sources = [
        KnowledgeSource(
            source_id="market_data_feed",
            name="市场数据源",
            source_type="market",
            reliability_score=0.9,
            update_frequency=1,
            data_quality=0.95,
            last_updated=datetime.now(),
            metadata={"provider": "Bloomberg", "type": "real-time"}
        ),
        KnowledgeSource(
            source_id="news_api",
            name="新闻API",
            source_type="news",
            reliability_score=0.8,
            update_frequency=2,
            data_quality=0.85,
            last_updated=datetime.now(),
            metadata={"provider": "Reuters", "language": "multi"}
        ),
        KnowledgeSource(
            source_id="technical_analysis",
            name="技术分析引擎",
            source_type="technical",
            reliability_score=0.85,
            update_frequency=4,
            data_quality=0.9,
            last_updated=datetime.now(),
            metadata={"algorithm": "ML-based", "accuracy": 0.87}
        )
    ]
    
    # 创建示例知识项
    items = [
        KnowledgeItem(
            item_id="item_001",
            content="市场数据显示主要指数上涨2.5%，交易量增加30%",
            source_id="market_data_feed",
            category="market_data",
            importance=0.9,
            freshness=0.95,
            confidence=0.92,
            timestamp=datetime.now() - timedelta(minutes=30),
            tags=["market", "index", "volume"],
            metadata={"symbol": "SPY", "change_percent": 2.5}
        ),
        KnowledgeItem(
            item_id="item_002",
            content="央行发布重要声明，暗示可能调整利率政策",
            source_id="news_api",
            category="monetary_policy",
            importance=0.95,
            freshness=0.88,
            confidence=0.85,
            timestamp=datetime.now() - timedelta(hours=1),
            tags=["central_bank", "interest_rate", "policy"],
            metadata={"source_confidence": "high", "verified": True}
        ),
        KnowledgeItem(
            item_id="item_003",
            content="技术指标显示RSI超买信号，建议谨慎操作",
            source_id="technical_analysis",
            category="technical_signal",
            importance=0.7,
            freshness=0.75,
            confidence=0.8,
            timestamp=datetime.now() - timedelta(hours=2),
            tags=["RSI", "overbought", "technical"],
            metadata={"indicator": "RSI", "value": 78.5, "threshold": 70}
        ),
        KnowledgeItem(
            item_id="item_004",
            content="机构投资者情绪转向乐观，买入信号增强",
            source_id="news_api",
            category="sentiment",
            importance=0.6,
            freshness=0.7,
            confidence=0.75,
            timestamp=datetime.now() - timedelta(hours=3),
            tags=["institutional", "sentiment", "buying"],
            metadata={"survey_size": 500, "confidence_interval": 0.05}
        )
    ]
    
    return sources, items


def main():
    """主函数 - 演示知识状态聚合器功能"""
    print("=" * 60)
    print("C9 知识状态聚合器演示")
    print("=" * 60)
    
    # 创建聚合器实例
    aggregator = KnowledgeStateAggregator()
    
    # 创建示例数据
    sources, items = create_sample_data()
    
    # 注册数据源
    print("\n1. 注册数据源...")
    for source in sources:
        aggregator.register_source(source)
        print(f"   ✓ {source.name} (可靠性: {source.reliability_score:.2f})")
    
    # 添加知识项
    print("\n2. 添加知识项...")
    for item in items:
        aggregator.add_knowledge_item(item)
        print(f"   ✓ {item.item_id}: {item.content[:50]}...")
    
    # 处理知识状态
    print("\n3. 处理知识状态...")
    current_state = aggregator.process_knowledge()
    print(f"   ✓ 处理完成")
    print(f"   • 总体质量: {current_state.overall_quality:.2f}")
    print(f"   • 覆盖度: {current_state.coverage_score:.2f}")
    print(f"   • 一致性: {current_state.coherence_score:.2f}")
    print(f"   • 时效性: {current_state.freshness_score:.2f}")
    print(f"   • 可靠性: {current_state.reliability_score:.2f}")
    
    # 生成报告
    print("\n4. 生成知识状态报告...")
    summary_report = aggregator.generate_report('summary')
    print(f"   ✓ 摘要报告生成完成")
    print(f"   • 知识项总数: {summary_report['knowledge_summary']['total_items']}")
    print(f"   • 数据类别: {summary_report['knowledge_summary']['categories']}")
    print(f"   • 预警数量: {len(summary_report['alerts'])}")
    
    # 详细报告
    detailed_report = aggregator.generate_report('detailed')
    print(f"   ✓ 详细报告生成完成")
    
    # 预警报告
    alert_report = aggregator.generate_report('alert')
    print(f"   ✓ 预警报告生成完成")
    
    # 保存报告
    print("\n5. 保存报告...")
    with open('knowledge_state_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2, default=str)
    print("   ✓ 摘要报告已保存到 knowledge_state_summary.json")
    
    with open('knowledge_state_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_report, f, ensure_ascii=False, indent=2, default=str)
    print("   ✓ 详细报告已保存到 knowledge_state_detailed.json")
    
    with open('knowledge_state_alerts.json', 'w', encoding='utf-8') as f:
        json.dump(alert_report, f, ensure_ascii=False, indent=2, default=str)
    print("   ✓ 预警报告已保存到 knowledge_state_alerts.json")
    
    # 导出状态
    print("\n6. 导出知识状态...")
    aggregator.export_state('knowledge_state_export.json')
    print("   ✓ 知识状态已导出到 knowledge_state_export.json")
    
    # 启动监控（演示用，5秒后停止）
    print("\n7. 启动知识状态监控...")
    aggregator.start_monitoring(interval=5)
    print("   ✓ 监控已启动，5秒后自动停止...")
    
    time.sleep(5)
    aggregator.stop_monitoring()
    print("   ✓ 监控已停止")
    
    print("\n" + "=" * 60)
    print("演示完成！知识状态聚合器运行正常。")
    print("=" * 60)


if __name__ == "__main__":
    main()