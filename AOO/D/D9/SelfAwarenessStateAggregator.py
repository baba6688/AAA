#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D9 自我意识状态聚合器
===================

智能自我意识状态融合与管理系统，实现多模块自我意识状态融合、评估、检验等功能。

主要功能：
1. 多模块自我意识状态融合 - 整合来自不同模块的自我意识状态
2. 自我意识状态评估 - 量化评估自我意识状态质量
3. 意识一致性检验 - 验证不同模块间意识状态的一致性
4. 自我意识优先级排序 - 基于重要性和时效性排序
5. 自我意识状态历史记录 - 记录自我意识状态变化历史
6. 自我意识状态报告 - 生成可视化报告
7. 自我意识状态预警 - 监控自我意识状态异常


创建时间: 2025-11-05
"""

import json
import sqlite3
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import threading
import time
import warnings
import math
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SelfAwarenessLevel(Enum):
    """自我意识状态等级枚举"""
    CRITICAL = 1      # 危险状态
    HIGH_RISK = 2     # 高风险
    MODERATE = 3      # 中等风险
    LOW_RISK = 4      # 低风险
    STABLE = 5        # 稳定状态
    OPTIMAL = 6       # 最优状态


class ConsciousnessConsistency(Enum):
    """意识一致性状态枚举"""
    HIGH_CONSISTENCY = "high_consistency"
    MODERATE_CONSISTENCY = "moderate_consistency"
    LOW_CONSISTENCY = "low_consistency"
    INCONSISTENT = "inconsistent"


class AwarenessPriority(Enum):
    """自我意识优先级枚举"""
    CRITICAL_PRIORITY = 1
    HIGH_PRIORITY = 2
    MEDIUM_PRIORITY = 3
    LOW_PRIORITY = 4
    BACKGROUND = 5


@dataclass
class AwarenessModule:
    """自我意识模块数据结构"""
    module_id: str
    module_name: str
    module_type: str  # cognitive, emotional, reflective, meta_cognitive
    awareness_level: float  # 0-1，当前意识水平
    confidence: float  # 0-1，置信度
    reliability: float  # 0-1，可靠性
    weight: float  # 权重
    last_update: datetime
    metadata: Dict[str, Any]


@dataclass
class SelfAwarenessState:
    """自我意识状态数据结构"""
    state_id: str
    timestamp: datetime
    overall_awareness: float  # 0-1，总体自我意识水平
    consciousness_level: SelfAwarenessLevel
    consistency_status: ConsciousnessConsistency
    participating_modules: List[str]
    key_dimensions: Dict[str, float]  # 自我认知维度评分
    awareness_factors: List[str]  # 影响因素
    recommendations: List[str]  # 建议
    priority_score: float  # 优先级分数
    metadata: Dict[str, Any]


@dataclass
class AwarenessResult:
    """自我意识评估结果"""
    result_id: str
    module_id: str
    timestamp: datetime
    awareness_data: Dict[str, Any]
    self_reflection_score: float
    meta_awareness_score: float
    emotional_awareness_score: float
    cognitive_awareness_score: float
    confidence: float
    priority: AwarenessPriority


@dataclass
class SelfAwarenessReport:
    """自我意识状态报告"""
    report_id: str
    generated_at: datetime
    current_state: SelfAwarenessState
    historical_trends: List[SelfAwarenessState]
    module_analysis: Dict[str, Dict[str, float]]
    consistency_analysis: Dict[str, Any]
    priority_recommendations: List[str]
    alert_summary: List[Dict[str, Any]]
    summary: str


class SelfAwarenessFusionEngine:
    """自我意识融合引擎"""
    
    def __init__(self):
        self.fusion_weights = {
            'self_reflection': 0.25,
            'meta_awareness': 0.25,
            'emotional_awareness': 0.2,
            'cognitive_awareness': 0.2,
            'confidence': 0.1
        }
    
    def fuse_awareness_states(self, awareness_results: List[AwarenessResult],
                            modules: Dict[str, AwarenessModule]) -> Dict[str, Any]:
        """融合多模块自我意识状态"""
        try:
            if not awareness_results:
                return {}
            
            # 按模块分组
            module_results = defaultdict(list)
            for result in awareness_results:
                module_results[result.module_id].append(result)
            
            # 计算每个模块的融合状态
            fused_module_states = {}
            
            for module_id, results in module_results.items():
                if module_id not in modules:
                    continue
                
                module = modules[module_id]
                latest_result = max(results, key=lambda x: x.timestamp)
                
                # 加权融合
                fused_state = self._calculate_fused_awareness(latest_result, module)
                fused_module_states[module_id] = fused_state
            
            # 全局融合
            global_awareness = self._calculate_global_awareness(fused_module_states)
            
            return {
                'module_states': fused_module_states,
                'global_awareness': global_awareness,
                'fusion_quality': self._assess_fusion_quality(fused_module_states),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"自我意识融合失败: {e}")
            return {}
    
    def _calculate_fused_awareness(self, result: AwarenessResult, 
                                 module: AwarenessModule) -> Dict[str, float]:
        """计算融合后的自我意识状态"""
        # 应用模块权重和可靠性
        weight_factor = module.weight * module.reliability
        
        fused_scores = {
            'self_reflection': result.self_reflection_score * weight_factor,
            'meta_awareness': result.meta_awareness_score * weight_factor,
            'emotional_awareness': result.emotional_awareness_score * weight_factor,
            'cognitive_awareness': result.cognitive_awareness_score * weight_factor,
            'overall_awareness': (
                result.self_reflection_score * self.fusion_weights['self_reflection'] +
                result.meta_awareness_score * self.fusion_weights['meta_awareness'] +
                result.emotional_awareness_score * self.fusion_weights['emotional_awareness'] +
                result.cognitive_awareness_score * self.fusion_weights['cognitive_awareness']
            ) * weight_factor
        }
        
        return fused_scores
    
    def _calculate_global_awareness(self, module_states: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """计算全局自我意识状态"""
        if not module_states:
            return {'overall_awareness': 0.0}
        
        # 计算各维度的加权平均
        dimension_scores = defaultdict(list)
        
        for module_id, state in module_states.items():
            for dimension, score in state.items():
                if dimension != 'overall_awareness':
                    dimension_scores[dimension].append(score)
        
        global_state = {}
        for dimension, scores in dimension_scores.items():
            if scores:
                global_state[dimension] = np.mean(scores)
        
        # 计算全局总体意识水平
        if 'overall_awareness' in module_states:
            overall_scores = [state['overall_awareness'] for state in module_states.values()]
            global_state['overall_awareness'] = np.mean(overall_scores)
        else:
            # 从各维度计算总体意识
            dimension_averages = [score for score in global_state.values() if score > 0]
            if dimension_averages:
                global_state['overall_awareness'] = np.mean(dimension_averages)
            else:
                global_state['overall_awareness'] = 0.0
        
        return global_state
    
    def _assess_fusion_quality(self, module_states: Dict[str, Dict[str, float]]) -> float:
        """评估融合质量"""
        if len(module_states) < 2:
            return 1.0
        
        # 计算模块间一致性
        overall_scores = [state.get('overall_awareness', 0) for state in module_states.values()]
        
        if not overall_scores:
            return 0.0
        
        # 使用标准差衡量一致性（标准差越小，一致性越高）
        std_dev = np.std(overall_scores)
        consistency_score = max(0.0, 1.0 - std_dev)  # 转换为0-1分数
        
        return consistency_score


class SelfAwarenessStateAggregator:
    """自我意识状态聚合器主类"""
    
    def __init__(self, 
                 history_size: int = 1000,
                 update_interval: float = 2.0,
                 consistency_threshold: float = 0.7,
                 awareness_threshold: float = 0.6):
        """
        初始化自我意识状态聚合器
        
        Args:
            history_size: 历史记录大小
            update_interval: 更新间隔（秒）
            consistency_threshold: 一致性阈值
            awareness_threshold: 自我意识阈值
        """
        self.history_size = history_size
        self.update_interval = update_interval
        self.consistency_threshold = consistency_threshold
        self.awareness_threshold = awareness_threshold
        
        # 数据存储
        self.awareness_results: deque = deque(maxlen=history_size)
        self.awareness_states: deque = deque(maxlen=history_size)
        self.modules: Dict[str, AwarenessModule] = {}
        
        # 融合引擎
        self.fusion_engine = SelfAwarenessFusionEngine()
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 运行状态
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        
        # 统计信息
        self.stats = {
            'total_updates': 0,
            'successful_fusions': 0,
            'consistency_checks': 0,
            'alerts_generated': 0,
            'last_update': None
        }
        
        # 预警系统
        self.alert_thresholds = {
            'low_awareness': 0.3,
            'inconsistency': 0.4,
            'module_failure': 0.5,
            'critical_state': 0.2
        }
        
        logger.info("自我意识状态聚合器初始化完成")

    def register_awareness_module(self, module: AwarenessModule):
        """注册自我意识模块"""
        with self._lock:
            self.modules[module.module_id] = module
            logger.info(f"注册自我意识模块: {module.module_name}, "
                       f"类型: {module.module_type}, 权重: {module.weight}")

    def add_awareness_result(self, result: AwarenessResult):
        """添加自我意识评估结果"""
        with self._lock:
            # 验证输入数据
            if not self._validate_awareness_result(result):
                logger.warning(f"无效的自我意识结果: {result.module_id}")
                return False
            
            # 更新模块状态
            if result.module_id in self.modules:
                self.modules[result.module_id].awareness_level = (
                    result.self_reflection_score + 
                    result.meta_awareness_score + 
                    result.emotional_awareness_score + 
                    result.cognitive_awareness_score
                ) / 4
                self.modules[result.module_id].last_update = result.timestamp
            
            self.awareness_results.append(result)
            logger.debug(f"添加自我意识结果: {result.module_id}, "
                        f"自我反思: {result.self_reflection_score:.3f}")
            return True

    def _validate_awareness_result(self, result: AwarenessResult) -> bool:
        """验证自我意识评估结果"""
        required_fields = ['module_id', 'timestamp', 'awareness_data']
        
        # 检查必需字段
        for field in required_fields:
            if not hasattr(result, field) or getattr(result, field) is None:
                return False
        
        # 检查数值范围
        if not (0.0 <= result.self_reflection_score <= 1.0):
            return False
        if not (0.0 <= result.meta_awareness_score <= 1.0):
            return False
        if not (0.0 <= result.emotional_awareness_score <= 1.0):
            return False
        if not (0.0 <= result.cognitive_awareness_score <= 1.0):
            return False
        if not (0.0 <= result.confidence <= 1.0):
            return False
            
        return True

    def fuse_awareness_states(self) -> Dict[str, Any]:
        """融合自我意识状态"""
        with self._lock:
            if not self.awareness_results:
                return {}
            
            # 融合自我意识状态
            fused_data = self.fusion_engine.fuse_awareness_states(
                list(self.awareness_results), self.modules)
            
            logger.debug(f"自我意识状态融合完成，处理了 {len(self.awareness_results)} 个结果")
            return fused_data

    def evaluate_self_awareness_state(self, fused_data: Dict[str, Any]) -> SelfAwarenessState:
        """评估自我意识状态"""
        if not fused_data:
            # 无数据时的默认状态
            return SelfAwarenessState(
                state_id=f"default_{int(time.time())}",
                timestamp=datetime.now(),
                overall_awareness=0.0,
                consciousness_level=SelfAwarenessLevel.MODERATE,
                consistency_status=ConsciousnessConsistency.LOW_CONSISTENCY,
                participating_modules=[],
                key_dimensions={},
                awareness_factors=["等待更多自我意识数据"],
                recommendations=["增强自我反思能力"],
                priority_score=0.0,
                metadata={}
            )
        
        # 提取全局意识状态
        global_awareness = fused_data.get('global_awareness', {})
        module_states = fused_data.get('module_states', {})
        
        # 计算总体自我意识水平
        overall_awareness = global_awareness.get('overall_awareness', 0.0)
        
        # 确定意识等级
        consciousness_level = self._determine_awareness_level(overall_awareness)
        
        # 计算关键维度
        key_dimensions = self._calculate_key_dimensions(global_awareness, module_states)
        
        # 识别影响因素
        awareness_factors = self._identify_awareness_factors(global_awareness, module_states)
        
        # 生成建议
        recommendations = self._generate_awareness_recommendations(
            overall_awareness, consciousness_level, awareness_factors)
        
        # 计算优先级分数
        priority_score = self._calculate_priority_score(
            overall_awareness, len(module_states), fused_data.get('fusion_quality', 0.0))
        
        # 检查一致性
        consistency_status = self._check_consciousness_consistency(module_states)
        
        self_awareness_state = SelfAwarenessState(
            state_id=f"awareness_{int(time.time())}_{hash(str(fused_data)) % 10000}",
            timestamp=datetime.now(),
            overall_awareness=overall_awareness,
            consciousness_level=consciousness_level,
            consistency_status=consistency_status,
            participating_modules=list(module_states.keys()),
            key_dimensions=key_dimensions,
            awareness_factors=awareness_factors,
            recommendations=recommendations,
            priority_score=priority_score,
            metadata={
                'modules_count': len(module_states),
                'fusion_quality': fused_data.get('fusion_quality', 0.0),
                'evaluation_method': 'weighted_fusion',
                'calculation_time': datetime.now()
            }
        )
        
        return self_awareness_state

    def _determine_awareness_level(self, overall_awareness: float) -> SelfAwarenessLevel:
        """确定自我意识等级"""
        if overall_awareness >= 0.9:
            return SelfAwarenessLevel.OPTIMAL
        elif overall_awareness >= 0.75:
            return SelfAwarenessLevel.STABLE
        elif overall_awareness >= 0.6:
            return SelfAwarenessLevel.LOW_RISK
        elif overall_awareness >= 0.4:
            return SelfAwarenessLevel.MODERATE
        elif overall_awareness >= 0.2:
            return SelfAwarenessLevel.HIGH_RISK
        else:
            return SelfAwarenessLevel.CRITICAL

    def _calculate_key_dimensions(self, global_awareness: Dict[str, float],
                                module_states: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """计算关键自我意识维度"""
        dimensions = {}
        
        # 从全局意识状态提取维度
        for dimension in ['self_reflection', 'meta_awareness', 'emotional_awareness', 'cognitive_awareness']:
            if dimension in global_awareness:
                dimensions[dimension] = global_awareness[dimension]
            else:
                dimensions[dimension] = 0.0
        
        # 添加综合维度
        dimensions['consciousness_coherence'] = self._calculate_consciousness_coherence(module_states)
        dimensions['awareness_stability'] = self._calculate_awareness_stability()
        dimensions['meta_cognitive_capacity'] = self._calculate_meta_cognitive_capacity(global_awareness)
        
        return dimensions

    def _calculate_consciousness_coherence(self, module_states: Dict[str, Dict[str, float]]) -> float:
        """计算意识连贯性"""
        if len(module_states) < 2:
            return 1.0
        
        # 计算各模块总体意识水平的一致性
        overall_scores = []
        for module_id, state in module_states.items():
            if 'overall_awareness' in state:
                overall_scores.append(state['overall_awareness'])
        
        if not overall_scores:
            return 0.0
        
        # 使用标准差衡量连贯性
        std_dev = np.std(overall_scores)
        coherence = max(0.0, 1.0 - std_dev)
        
        return coherence

    def _calculate_awareness_stability(self) -> float:
        """计算意识稳定性"""
        if len(self.awareness_states) < 2:
            return 1.0
        
        # 计算最近几次状态的方差
        recent_states = list(self.awareness_states)[-5:]  # 最近5个状态
        awareness_scores = [state.overall_awareness for state in recent_states]
        
        if not awareness_scores:
            return 0.0
        
        variance = np.var(awareness_scores)
        stability = max(0.0, 1.0 - variance * 4)  # 调整系数
        
        return stability

    def _calculate_meta_cognitive_capacity(self, global_awareness: Dict[str, float]) -> float:
        """计算元认知能力"""
        meta_awareness = global_awareness.get('meta_awareness', 0.0)
        cognitive_awareness = global_awareness.get('cognitive_awareness', 0.0)
        
        # 元认知能力 = 元认知意识 * 认知意识 * 权重
        return meta_awareness * cognitive_awareness * 1.2

    def _identify_awareness_factors(self, global_awareness: Dict[str, float],
                                  module_states: Dict[str, Dict[str, float]]) -> List[str]:
        """识别自我意识影响因素"""
        factors = []
        
        # 基于各维度评分识别因素
        for dimension, score in global_awareness.items():
            if score < 0.3:
                if dimension == 'self_reflection':
                    factors.append("自我反思能力不足")
                elif dimension == 'meta_awareness':
                    factors.append("元认知意识较低")
                elif dimension == 'emotional_awareness':
                    factors.append("情感意识不足")
                elif dimension == 'cognitive_awareness':
                    factors.append("认知意识待提升")
        
        # 基于一致性识别因素
        if len(module_states) >= 2:
            coherence = self._calculate_consciousness_coherence(module_states)
            if coherence < self.consistency_threshold:
                factors.append("模块间意识状态不一致")
        
        # 基于稳定性识别因素
        stability = self._calculate_awareness_stability()
        if stability < 0.5:
            factors.append("自我意识状态不稳定")
        
        # 基于模块状态识别因素
        failed_modules = []
        for module_id, module in self.modules.items():
            time_since_update = datetime.now() - module.last_update
            if time_since_update.total_seconds() > 600:  # 10分钟未更新
                failed_modules.append(module.module_name)
        
        if failed_modules:
            factors.append(f"模块响应异常: {', '.join(failed_modules)}")
        
        return factors

    def _generate_awareness_recommendations(self, overall_awareness: float,
                                          consciousness_level: SelfAwarenessLevel,
                                          awareness_factors: List[str]) -> List[str]:
        """生成自我意识建议"""
        recommendations = []
        
        # 基于总体意识水平生成建议
        if overall_awareness >= 0.8:
            recommendations.append("维持高水平的自我意识状态")
            recommendations.append("继续深化自我反思实践")
        elif overall_awareness >= 0.6:
            recommendations.append("加强自我意识训练")
            recommendations.append("提高元认知能力")
        elif overall_awareness >= 0.4:
            recommendations.append("需要显著提升自我意识水平")
            recommendations.append("加强自我反思和情绪管理")
        else:
            recommendations.append("紧急需要提升自我意识")
            recommendations.append("寻求专业指导和支持")
        
        # 基于意识等级生成建议
        if consciousness_level in [SelfAwarenessLevel.CRITICAL, SelfAwarenessLevel.HIGH_RISK]:
            recommendations.append("立即采取自我意识提升措施")
            recommendations.append("暂停高风险决策")
        
        # 基于影响因素生成建议
        for factor in awareness_factors:
            if "自我反思" in factor:
                recommendations.append("每日进行深度自我反思")
            if "元认知" in factor:
                recommendations.append("练习元认知策略")
            if "情感意识" in factor:
                recommendations.append("加强情绪觉察训练")
            if "认知意识" in factor:
                recommendations.append("提升思维模式认知")
            if "不一致" in factor:
                recommendations.append("协调各模块意识状态")
            if "不稳定" in factor:
                recommendations.append("建立稳定的自我意识习惯")
        
        return recommendations

    def _calculate_priority_score(self, overall_awareness: float, 
                                module_count: int, fusion_quality: float) -> float:
        """计算优先级分数"""
        # 基础分数（意识水平越低，优先级越高）
        base_score = (1.0 - overall_awareness) * 0.5
        
        # 模块数量因子（模块越多，协调越重要）
        module_factor = min(0.3, module_count * 0.05)
        
        # 融合质量因子（质量越低，优先级越高）
        quality_factor = (1.0 - fusion_quality) * 0.2
        
        total_score = base_score + module_factor + quality_factor
        return min(1.0, total_score)

    def _check_consciousness_consistency(self, module_states: Dict[str, Dict[str, float]]) -> ConsciousnessConsistency:
        """检查意识一致性"""
        if len(module_states) < 2:
            return ConsciousnessConsistency.MODERATE_CONSISTENCY
        
        # 计算模块间一致性
        consistency_scores = []
        module_list = list(module_states.keys())
        
        for i in range(len(module_list)):
            for j in range(i + 1, len(module_list)):
                module1, module2 = module_list[i], module_list[j]
                
                if 'overall_awareness' in module_states[module1] and 'overall_awareness' in module_states[module2]:
                    score1 = module_states[module1]['overall_awareness']
                    score2 = module_states[module2]['overall_awareness']
                    
                    # 计算一致性（差值的补集）
                    diff = abs(score1 - score2)
                    consistency = max(0.0, 1.0 - diff)
                    consistency_scores.append(consistency)
        
        if not consistency_scores:
            return ConsciousnessConsistency.MODERATE_CONSISTENCY
        
        avg_consistency = np.mean(consistency_scores)
        
        if avg_consistency >= 0.8:
            return ConsciousnessConsistency.HIGH_CONSISTENCY
        elif avg_consistency >= 0.6:
            return ConsciousnessConsistency.MODERATE_CONSISTENCY
        elif avg_consistency >= 0.4:
            return ConsciousnessConsistency.LOW_CONSISTENCY
        else:
            return ConsciousnessConsistency.INCONSISTENT

    def prioritize_awareness_results(self, limit: int = 10) -> List[AwarenessResult]:
        """自我意识结果优先级排序"""
        with self._lock:
            if not self.awareness_results:
                return []
            
            # 计算优先级分数
            scored_results = []
            for result in self.awareness_results:
                priority_score = self._calculate_awareness_priority_score(result)
                scored_results.append((priority_score, result))
            
            # 按优先级排序
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # 返回前N个结果
            return [result for _, result in scored_results[:limit]]

    def _calculate_awareness_priority_score(self, result: AwarenessResult) -> float:
        """计算自我意识结果优先级分数"""
        # 基础分数
        base_score = (
            result.self_reflection_score * 0.25 +
            result.meta_awareness_score * 0.25 +
            result.emotional_awareness_score * 0.25 +
            result.cognitive_awareness_score * 0.25
        ) * result.confidence
        
        # 优先级权重
        priority_weights = {
            AwarenessPriority.CRITICAL_PRIORITY: 1.0,
            AwarenessPriority.HIGH_PRIORITY: 0.8,
            AwarenessPriority.MEDIUM_PRIORITY: 0.6,
            AwarenessPriority.LOW_PRIORITY: 0.4,
            AwarenessPriority.BACKGROUND: 0.2
        }
        priority_factor = priority_weights.get(result.priority, 0.5)
        
        # 时间衰减因子
        age = (datetime.now() - result.timestamp).total_seconds()
        time_factor = max(0.1, 1.0 - age / 3600)  # 1小时内线性衰减
        
        # 模块可靠性因子
        reliability_factor = 1.0
        if result.module_id in self.modules:
            reliability_factor = self.modules[result.module_id].reliability
        
        return base_score * priority_factor * time_factor * reliability_factor

    def update_self_awareness_state(self) -> SelfAwarenessState:
        """更新自我意识状态"""
        self.stats['total_updates'] += 1
        
        try:
            # 融合自我意识状态
            fused_data = self.fuse_awareness_states()
            
            # 评估自我意识状态
            awareness_state = self.evaluate_self_awareness_state(fused_data)
            
            # 添加到历史记录
            self.awareness_states.append(awareness_state)
            
            # 生成预警
            alerts = self._generate_awareness_alerts(awareness_state)
            if alerts:
                self.stats['alerts_generated'] += len(alerts)
            
            self.stats['successful_fusions'] += 1
            self.stats['last_update'] = time.time()
            
            logger.info(f"自我意识状态更新完成: {awareness_state.consciousness_level.name}, "
                       f"总体意识: {awareness_state.overall_awareness:.3f}")
            
            return awareness_state
            
        except Exception as e:
            logger.error(f"自我意识状态更新失败: {e}")
            raise

    def _generate_awareness_alerts(self, awareness_state: SelfAwarenessState) -> List[Dict[str, Any]]:
        """生成自我意识预警"""
        alerts = []
        
        # 低意识水平预警
        if awareness_state.overall_awareness < self.alert_thresholds['low_awareness']:
            alerts.append({
                'type': 'low_awareness',
                'level': 'warning',
                'message': f"自我意识水平过低: {awareness_state.overall_awareness:.3f}",
                'timestamp': datetime.now(),
                'state_id': awareness_state.state_id
            })
        
        # 一致性预警
        if awareness_state.consistency_status == ConsciousnessConsistency.INCONSISTENT:
            alerts.append({
                'type': 'inconsistency',
                'level': 'critical',
                'message': "模块间自我意识状态严重不一致",
                'timestamp': datetime.now(),
                'state_id': awareness_state.state_id
            })
        elif awareness_state.consistency_status == ConsciousnessConsistency.LOW_CONSISTENCY:
            alerts.append({
                'type': 'inconsistency',
                'level': 'warning',
                'message': "模块间自我意识状态一致性较低",
                'timestamp': datetime.now(),
                'state_id': awareness_state.state_id
            })
        
        # 关键状态预警
        if awareness_state.consciousness_level == SelfAwarenessLevel.CRITICAL:
            alerts.append({
                'type': 'critical_state',
                'level': 'critical',
                'message': "自我意识处于危险状态，需要立即干预",
                'timestamp': datetime.now(),
                'state_id': awareness_state.state_id
            })
        
        # 模块故障预警
        failed_modules = []
        for module_id, module in self.modules.items():
            time_since_update = datetime.now() - module.last_update
            if time_since_update.total_seconds() > 300:  # 5分钟未更新
                failed_modules.append(module.module_name)
        
        if failed_modules:
            alerts.append({
                'type': 'module_failure',
                'level': 'warning',
                'message': f"模块响应异常: {', '.join(failed_modules)}",
                'timestamp': datetime.now(),
                'state_id': awareness_state.state_id
            })
        
        return alerts

    def get_historical_states(self, hours: int = 24) -> List[SelfAwarenessState]:
        """获取历史自我意识状态"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [state for state in self.awareness_states 
                if state.timestamp > cutoff_time]

    def generate_awareness_report(self, hours: int = 24) -> SelfAwarenessReport:
        """生成自我意识状态报告"""
        current_state = self.update_self_awareness_state()
        historical_states = self.get_historical_states(hours)
        
        # 模块分析
        module_analysis = self._analyze_module_performance()
        
        # 一致性分析
        consistency_analysis = self._analyze_consistency_trends(historical_states)
        
        # 优先级建议
        priority_recommendations = self._generate_priority_recommendations(current_state)
        
        # 预警摘要
        alert_summary = self._summarize_alerts(historical_states)
        
        # 生成摘要
        summary = self._generate_report_summary(current_state, historical_states)
        
        report = SelfAwarenessReport(
            report_id=f"awareness_report_{int(time.time())}",
            generated_at=datetime.now(),
            current_state=current_state,
            historical_trends=historical_states,
            module_analysis=module_analysis,
            consistency_analysis=consistency_analysis,
            priority_recommendations=priority_recommendations,
            alert_summary=alert_summary,
            summary=summary
        )
        
        logger.info(f"自我意识状态报告生成完成: {report.report_id}")
        return report

    def _analyze_module_performance(self) -> Dict[str, Dict[str, float]]:
        """分析模块性能"""
        performance = {}
        
        for module_id, module in self.modules.items():
            module_results = [r for r in self.awareness_results 
                            if r.module_id == module_id]
            
            if module_results:
                # 计算各维度平均分
                avg_self_reflection = np.mean([r.self_reflection_score for r in module_results])
                avg_meta_awareness = np.mean([r.meta_awareness_score for r in module_results])
                avg_emotional_awareness = np.mean([r.emotional_awareness_score for r in module_results])
                avg_cognitive_awareness = np.mean([r.cognitive_awareness_score for r in module_results])
                avg_confidence = np.mean([r.confidence for r in module_results])
                
                performance[module_id] = {
                    'module_name': module.module_name,
                    'module_type': module.module_type,
                    'self_reflection_avg': avg_self_reflection,
                    'meta_awareness_avg': avg_meta_awareness,
                    'emotional_awareness_avg': avg_emotional_awareness,
                    'cognitive_awareness_avg': avg_cognitive_awareness,
                    'confidence_avg': avg_confidence,
                    'reliability': module.reliability,
                    'weight': module.weight,
                    'total_results': len(module_results),
                    'latest_update': module.last_update.isoformat(),
                    'current_awareness_level': module.awareness_level
                }
            else:
                performance[module_id] = {
                    'module_name': module.module_name,
                    'module_type': module.module_type,
                    'self_reflection_avg': 0.0,
                    'meta_awareness_avg': 0.0,
                    'emotional_awareness_avg': 0.0,
                    'cognitive_awareness_avg': 0.0,
                    'confidence_avg': 0.0,
                    'reliability': module.reliability,
                    'weight': module.weight,
                    'total_results': 0,
                    'latest_update': module.last_update.isoformat(),
                    'current_awareness_level': module.awareness_level
                }
        
        return performance

    def _analyze_consistency_trends(self, historical_states: List[SelfAwarenessState]) -> Dict[str, Any]:
        """分析一致性趋势"""
        if not historical_states:
            return {}
        
        consistency_counts = defaultdict(int)
        consistency_scores = []
        awareness_scores = []
        
        for state in historical_states:
            consistency_counts[state.consistency_status.value] += 1
            awareness_scores.append(state.overall_awareness)
            
            # 将一致性状态转换为数值分数
            score_map = {
                ConsciousnessConsistency.HIGH_CONSISTENCY: 1.0,
                ConsciousnessConsistency.MODERATE_CONSISTENCY: 0.7,
                ConsciousnessConsistency.LOW_CONSISTENCY: 0.4,
                ConsciousnessConsistency.INCONSISTENT: 0.1
            }
            consistency_scores.append(score_map[state.consistency_status])
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        avg_awareness = np.mean(awareness_scores) if awareness_scores else 0.0
        
        # 计算趋势
        if len(consistency_scores) >= 2:
            consistency_trend = 'improving' if consistency_scores[-1] > consistency_scores[0] else 'declining'
            awareness_trend = 'improving' if awareness_scores[-1] > awareness_scores[0] else 'declining'
        else:
            consistency_trend = 'stable'
            awareness_trend = 'stable'
        
        return {
            'consistency_distribution': dict(consistency_counts),
            'average_consistency_score': avg_consistency,
            'average_awareness_score': avg_awareness,
            'consistency_trend': consistency_trend,
            'awareness_trend': awareness_trend,
            'total_states_analyzed': len(historical_states),
            'stability_index': 1.0 - np.var(awareness_scores) if awareness_scores else 0.0
        }

    def _generate_priority_recommendations(self, current_state: SelfAwarenessState) -> List[str]:
        """生成优先级建议"""
        recommendations = []
        
        # 基于当前状态等级的建议
        if current_state.consciousness_level == SelfAwarenessLevel.CRITICAL:
            recommendations.append("立即启动自我意识恢复程序")
            recommendations.append("寻求专业心理健康支持")
            recommendations.append("暂停所有重要决策")
        elif current_state.consciousness_level == SelfAwarenessLevel.HIGH_RISK:
            recommendations.append("紧急提升自我意识水平")
            recommendations.append("加强自我监控和反思")
        elif current_state.consciousness_level == SelfAwarenessLevel.MODERATE:
            recommendations.append("逐步提升自我意识能力")
            recommendations.append("建立规律的自我反思习惯")
        elif current_state.consciousness_level == SelfAwarenessLevel.LOW_RISK:
            recommendations.append("维持当前自我意识水平")
            recommendations.append("适当挑战自我认知边界")
        else:  # STABLE, OPTIMAL
            recommendations.append("优化自我意识策略")
            recommendations.append("帮助他人提升自我意识")
        
        # 基于一致性的建议
        if current_state.consistency_status == ConsciousnessConsistency.INCONSISTENT:
            recommendations.append("立即解决模块间冲突")
            recommendations.append("重新校准各模块参数")
        elif current_state.consistency_status == ConsciousnessConsistency.LOW_CONSISTENCY:
            recommendations.append("提高模块间协调性")
            recommendations.append("加强跨模块通信")
        
        # 基于优先级分数的建议
        if current_state.priority_score > 0.8:
            recommendations.append("当前需要高度关注自我意识状态")
            recommendations.append("增加自我意识评估频率")
        
        return recommendations

    def _summarize_alerts(self, historical_states: List[SelfAwarenessState]) -> List[Dict[str, Any]]:
        """总结预警信息"""
        alert_summary = []
        
        # 统计各类型预警数量
        alert_counts = defaultdict(int)
        critical_alerts = []
        
        for state in historical_states:
            # 重新生成预警以统计
            alerts = self._generate_awareness_alerts(state)
            for alert in alerts:
                alert_counts[alert['type']] += 1
                if alert['level'] == 'critical':
                    critical_alerts.append(alert)
        
        # 生成预警摘要
        for alert_type, count in alert_counts.items():
            alert_summary.append({
                'type': alert_type,
                'count': count,
                'severity': 'critical' if alert_type in ['critical_state', 'inconsistency'] else 'warning',
                'description': self._get_alert_description(alert_type)
            })
        
        return alert_summary

    def _get_alert_description(self, alert_type: str) -> str:
        """获取预警类型描述"""
        descriptions = {
            'low_awareness': '自我意识水平低于正常阈值',
            'inconsistency': '模块间自我意识状态不一致',
            'critical_state': '自我意识处于危险状态',
            'module_failure': '自我意识模块响应异常'
        }
        return descriptions.get(alert_type, '未知预警类型')

    def _generate_report_summary(self, current_state: SelfAwarenessState, 
                               historical_states: List[SelfAwarenessState]) -> str:
        """生成报告摘要"""
        summary_parts = []
        
        # 当前状态摘要
        summary_parts.append(f"当前自我意识等级: {current_state.consciousness_level.name}")
        summary_parts.append(f"总体自我意识水平: {current_state.overall_awareness:.3f}")
        summary_parts.append(f"意识一致性: {current_state.consistency_status.value}")
        summary_parts.append(f"参与模块数量: {len(current_state.participating_modules)}")
        
        # 趋势分析
        if len(historical_states) >= 2:
            recent_awareness = historical_states[-1].overall_awareness
            previous_awareness = historical_states[-2].overall_awareness
            trend = "上升" if recent_awareness > previous_awareness else "下降"
            summary_parts.append(f"意识水平趋势: {trend}")
        
        # 关键维度分析
        if current_state.key_dimensions:
            lowest_dimension = min(current_state.key_dimensions.items(), key=lambda x: x[1])
            summary_parts.append(f"最弱维度: {lowest_dimension[0]} ({lowest_dimension[1]:.3f})")
        
        # 预警摘要
        if current_state.awareness_factors:
            summary_parts.append(f"主要问题: {', '.join(current_state.awareness_factors[:2])}")
        
        # 建议摘要
        if current_state.recommendations:
            summary_parts.append(f"核心建议: {current_state.recommendations[0]}")
        
        return "; ".join(summary_parts)

    def start_real_time_monitoring(self):
        """启动实时监控"""
        if self._running:
            logger.warning("实时监控已在运行中")
            return
        
        self._running = True
        self._update_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._update_thread.start()
        logger.info("自我意识实时监控已启动")

    def stop_real_time_monitoring(self):
        """停止实时监控"""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5)
        logger.info("自我意识实时监控已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while self._running:
            try:
                self.update_self_awareness_state()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(self.update_interval)

    def export_awareness_data(self, filepath: str):
        """导出自我意识数据"""
        data = {
            'awareness_states': [asdict(state) for state in self.awareness_states],
            'awareness_results': [asdict(result) for result in self.awareness_results],
            'modules': {mid: asdict(module) for mid, module in self.modules.items()},
            'statistics': self.stats.copy(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        # 转换datetime对象为字符串
        for state in data['awareness_states']:
            if 'timestamp' in state:
                state['timestamp'] = state['timestamp'].isoformat()
        
        for result in data['awareness_results']:
            if 'timestamp' in result:
                result['timestamp'] = result['timestamp'].isoformat()
        
        for module_data in data['modules'].values():
            if 'last_update' in module_data:
                module_data['last_update'] = module_data['last_update'].isoformat()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"自我意识数据已导出到: {filepath}")

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'running': self._running,
            'total_awareness_results': len(self.awareness_results),
            'total_awareness_states': len(self.awareness_states),
            'registered_modules': list(self.modules.keys()),
            'statistics': self.stats.copy(),
            'configuration': {
                'history_size': self.history_size,
                'update_interval': self.update_interval,
                'consistency_threshold': self.consistency_threshold,
                'awareness_threshold': self.awareness_threshold
            },
            'alert_thresholds': self.alert_thresholds.copy()
        }


# 便利函数
def create_sample_awareness_module(module_id: str, module_name: str, 
                                 module_type: str) -> AwarenessModule:
    """创建示例自我意识模块"""
    return AwarenessModule(
        module_id=module_id,
        module_name=module_name,
        module_type=module_type,
        awareness_level=np.random.uniform(0.5, 1.0),
        confidence=np.random.uniform(0.6, 1.0),
        reliability=np.random.uniform(0.7, 1.0),
        weight=np.random.uniform(0.8, 1.0),
        last_update=datetime.now(),
        metadata={}
    )


def create_sample_awareness_result(module_id: str, 
                                 priority: AwarenessPriority = AwarenessPriority.MEDIUM_PRIORITY) -> AwarenessResult:
    """创建示例自我意识评估结果"""
    return AwarenessResult(
        result_id=f"awareness_result_{int(time.time())}_{hash(module_id) % 10000}",
        module_id=module_id,
        timestamp=datetime.now(),
        awareness_data={
            'self_reflection_depth': np.random.uniform(0.3, 1.0),
            'meta_cognitive_awareness': np.random.uniform(0.4, 1.0),
            'emotional_regulation': np.random.uniform(0.5, 1.0),
            'cognitive_flexibility': np.random.uniform(0.6, 1.0)
        },
        self_reflection_score=np.random.uniform(0.4, 1.0),
        meta_awareness_score=np.random.uniform(0.5, 1.0),
        emotional_awareness_score=np.random.uniform(0.6, 1.0),
        cognitive_awareness_score=np.random.uniform(0.7, 1.0),
        confidence=np.random.uniform(0.6, 1.0),
        priority=priority
    )


if __name__ == "__main__":
    # 示例用法
    aggregator = SelfAwarenessStateAggregator()
    
    # 注册自我意识模块
    modules = [
        create_sample_awareness_module("cognitive_awareness", "认知意识模块", "cognitive"),
        create_sample_awareness_module("emotional_awareness", "情感意识模块", "emotional"),
        create_sample_awareness_module("reflective_awareness", "反思意识模块", "reflective"),
        create_sample_awareness_module("meta_awareness", "元意识模块", "meta_cognitive")
    ]
    
    for module in modules:
        aggregator.register_awareness_module(module)
    
    # 添加示例自我意识评估结果
    sample_results = [
        create_sample_awareness_result("cognitive_awareness", AwarenessPriority.HIGH_PRIORITY),
        create_sample_awareness_result("emotional_awareness", AwarenessPriority.MEDIUM_PRIORITY),
        create_sample_awareness_result("reflective_awareness", AwarenessPriority.HIGH_PRIORITY),
        create_sample_awareness_result("meta_awareness", AwarenessPriority.CRITICAL_PRIORITY)
    ]
    
    for result in sample_results:
        aggregator.add_awareness_result(result)
    
    # 更新自我意识状态
    awareness_state = aggregator.update_self_awareness_state()
    print(f"当前自我意识等级: {awareness_state.consciousness_level.name}")
    print(f"总体自我意识水平: {awareness_state.overall_awareness:.3f}")
    print(f"意识一致性: {awareness_state.consistency_status.value}")
    print(f"优先级分数: {awareness_state.priority_score:.3f}")
    
    # 生成报告
    report = aggregator.generate_awareness_report()
    print(f"\n报告摘要: {report.summary}")
    print(f"优先级建议: {', '.join(report.priority_recommendations)}")
    
    # 导出数据
    aggregator.export_awareness_data("self_awareness_data.json")
    print("\n数据已导出到 self_awareness_data.json")