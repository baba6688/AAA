#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E9 创新状态聚合器
Innovation State Aggregator

功能：
1. 多模块创新状态融合
2. 创新状态评估
3. 创新一致性检验
4. 创新优先级排序
5. 创新状态历史记录
6. 创新状态报告
7. 创新状态预警


创建时间：2025-11-05
版本：1.0.0
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from collections import defaultdict, deque
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


class InnovationJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理枚举类型"""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class InnovationType(Enum):
    """创新类型枚举"""
    TECHNICAL = "技术创新"
    PRODUCT = "产品创新"
    PROCESS = "流程创新"
    BUSINESS = "商业模式创新"
    MARKET = "市场创新"
    ORGANIZATIONAL = "组织创新"


class InnovationStatus(Enum):
    """创新状态枚举"""
    CONCEPT = "概念阶段"
    DEVELOPMENT = "开发阶段"
    TESTING = "测试阶段"
    IMPLEMENTATION = "实施阶段"
    VALIDATION = "验证阶段"
    DEPLOYMENT = "部署阶段"
    MAINTENANCE = "维护阶段"


class PriorityLevel(Enum):
    """优先级级别"""
    CRITICAL = "关键"
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"


@dataclass
class InnovationMetrics:
    """创新指标数据结构"""
    novelty: float = 0.0  # 新颖性 (0-1)
    feasibility: float = 0.0  # 可行性 (0-1)
    impact: float = 0.0  # 影响度 (0-1)
    urgency: float = 0.0  # 紧急性 (0-1)
    resource_requirement: float = 0.0  # 资源需求 (0-1)
    risk_level: float = 0.0  # 风险水平 (0-1)
    market_potential: float = 0.0  # 市场潜力 (0-1)
    technical_maturity: float = 0.0  # 技术成熟度 (0-1)
    business_value: float = 0.0  # 商业价值 (0-1)
    scalability: float = 0.0  # 可扩展性 (0-1)


@dataclass
class InnovationState:
    """创新状态数据结构"""
    id: str
    name: str
    type: InnovationType
    status: InnovationStatus
    description: str
    metrics: InnovationMetrics
    priority: PriorityLevel
    created_at: datetime
    updated_at: datetime
    owner: str
    tags: List[str]
    dependencies: List[str]
    progress: float = 0.0  # 进度 (0-100)
    budget: float = 0.0  # 预算
    timeline: Dict[str, datetime] = None  # 时间线
    stakeholders: List[str] = None  # 利益相关者
    success_criteria: List[str] = None  # 成功标准
    risk_factors: List[str] = None  # 风险因素
    innovation_score: float = 0.0  # 综合创新分数


@dataclass
class ConsistencyResult:
    """一致性检验结果"""
    is_consistent: bool
    conflicts: List[str]
    synergies: List[str]
    recommendations: List[str]
    consistency_score: float


@dataclass
class PriorityResult:
    """优先级排序结果"""
    innovation_id: str
    priority_score: float
    priority_level: PriorityLevel
    ranking: int
    factors: Dict[str, float]


@dataclass
class HistoricalRecord:
    """历史记录数据结构"""
    timestamp: datetime
    innovation_id: str
    state: Dict[str, Any]
    metrics: InnovationMetrics
    events: List[str]
    performance_indicators: Dict[str, float]


class InnovationStateAggregator:
    """创新状态聚合器主类"""
    
    def __init__(self, db_path: str = "innovation_state.db"):
        """
        初始化创新状态聚合器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.logger = self._setup_logger()
        self._lock = threading.Lock()
        self._scaler = StandardScaler()
        self._innovation_cache = {}
        self._historical_data = deque(maxlen=10000)
        
        # 初始化数据库
        self._init_database()
        
        # 加载历史数据
        self._load_historical_data()
        
        self.logger.info("创新状态聚合器初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("InnovationStateAggregator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_database(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创新状态表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS innovation_states (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    description TEXT,
                    metrics TEXT,
                    priority TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    owner TEXT,
                    tags TEXT,
                    dependencies TEXT,
                    progress REAL DEFAULT 0.0,
                    budget REAL DEFAULT 0.0,
                    timeline TEXT,
                    stakeholders TEXT,
                    success_criteria TEXT,
                    risk_factors TEXT,
                    innovation_score REAL DEFAULT 0.0
                )
            ''')
            
            # 历史记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    innovation_id TEXT NOT NULL,
                    state TEXT,
                    metrics TEXT,
                    events TEXT,
                    performance_indicators TEXT,
                    FOREIGN KEY (innovation_id) REFERENCES innovation_states (id)
                )
            ''')
            
            # 预警记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    innovation_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (innovation_id) REFERENCES innovation_states (id)
                )
            ''')
            
            conn.commit()
    
    def _load_historical_data(self):
        """加载历史数据到内存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM historical_records 
                    ORDER BY timestamp DESC 
                    LIMIT 5000
                ''')
                
                for row in cursor.fetchall():
                    record = HistoricalRecord(
                        timestamp=datetime.fromisoformat(row[1]),
                        innovation_id=row[2],
                        state=json.loads(row[3]) if row[3] else {},
                        metrics=InnovationMetrics(**json.loads(row[4])) if row[4] else InnovationMetrics(),
                        events=json.loads(row[5]) if row[5] else [],
                        performance_indicators=json.loads(row[6]) if row[6] else {}
                    )
                    self._historical_data.append(record)
                    
        except Exception as e:
            self.logger.error(f"加载历史数据失败: {e}")
    
    def register_innovation(self, innovation: InnovationState) -> bool:
        """
        注册新的创新项目
        
        Args:
            innovation: 创新状态对象
            
        Returns:
            bool: 注册是否成功
        """
        try:
            with self._lock:
                # 计算创新分数
                innovation.innovation_score = self._calculate_innovation_score(innovation.metrics)
                
                # 保存到数据库
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO innovation_states 
                        (id, name, type, status, description, metrics, priority, 
                         created_at, updated_at, owner, tags, dependencies, 
                         progress, budget, timeline, stakeholders, success_criteria, 
                         risk_factors, innovation_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        innovation.id, innovation.name, innovation.type.value,
                        innovation.status.value, innovation.description,
                        json.dumps(asdict(innovation.metrics), cls=InnovationJSONEncoder),
                        innovation.priority.value, innovation.created_at.isoformat(),
                        innovation.updated_at.isoformat(), innovation.owner,
                        json.dumps(innovation.tags), json.dumps(innovation.dependencies),
                        innovation.progress, innovation.budget,
                        json.dumps(innovation.timeline) if innovation.timeline else None,
                        json.dumps(innovation.stakeholders) if innovation.stakeholders else None,
                        json.dumps(innovation.success_criteria) if innovation.success_criteria else None,
                        json.dumps(innovation.risk_factors) if innovation.risk_factors else None,
                        innovation.innovation_score
                    ))
                    conn.commit()
                
                # 缓存到内存
                self._innovation_cache[innovation.id] = innovation
                
                # 记录历史
                self._record_historical_event(innovation.id, "创新项目注册", innovation)
                
                self.logger.info(f"创新项目 {innovation.name} ({innovation.id}) 注册成功")
                return True
                
        except Exception as e:
            self.logger.error(f"注册创新项目失败: {e}")
            return False
    
    def _calculate_innovation_score(self, metrics: InnovationMetrics) -> float:
        """
        计算综合创新分数
        
        Args:
            metrics: 创新指标
            
        Returns:
            float: 创新分数 (0-100)
        """
        # 权重配置
        weights = {
            'novelty': 0.2,           # 新颖性
            'feasibility': 0.15,      # 可行性
            'impact': 0.2,            # 影响度
            'urgency': 0.1,           # 紧急性
            'market_potential': 0.15, # 市场潜力
            'technical_maturity': 0.1, # 技术成熟度
            'business_value': 0.1     # 商业价值
        }
        
        # 计算加权分数
        score = (
            metrics.novelty * weights['novelty'] +
            metrics.feasibility * weights['feasibility'] +
            metrics.impact * weights['impact'] +
            metrics.urgency * weights['urgency'] +
            metrics.market_potential * weights['market_potential'] +
            metrics.technical_maturity * weights['technical_maturity'] +
            metrics.business_value * weights['business_value']
        ) * 100
        
        return min(100.0, max(0.0, score))
    
    def fuse_innovation_states(self, innovation_ids: List[str]) -> Dict[str, Any]:
        """
        融合多个创新状态
        
        Args:
            innovation_ids: 创新项目ID列表
            
        Returns:
            Dict[str, Any]: 融合结果
        """
        try:
            innovations = []
            for innovation_id in innovation_ids:
                innovation = self.get_innovation_state(innovation_id)
                if innovation:
                    innovations.append(innovation)
            
            if not innovations:
                return {"error": "未找到有效的创新项目"}
            
            # 获取所有指标数据
            all_metrics = [innovation.metrics for innovation in innovations]
            
            # 融合指标
            fused_metrics = self._fuse_metrics(all_metrics)
            
            # 分析一致性
            consistency = self.check_innovation_consistency(innovation_ids)
            
            # 计算协同效应
            synergies = self._calculate_synergies(innovations)
            
            # 生成融合报告
            fusion_report = {
                "fused_metrics": asdict(fused_metrics),
                "consistency": asdict(consistency),
                "synergies": synergies,
                "innovation_count": len(innovations),
                "fusion_timestamp": datetime.now().isoformat(),
                "recommendations": self._generate_fusion_recommendations(innovations, consistency)
            }
            
            self.logger.info(f"成功融合 {len(innovations)} 个创新项目")
            return fusion_report
            
        except Exception as e:
            self.logger.error(f"创新状态融合失败: {e}")
            return {"error": str(e)}
    
    def _fuse_metrics(self, metrics_list: List[InnovationMetrics]) -> InnovationMetrics:
        """
        融合多个指标
        
        Args:
            metrics_list: 指标列表
            
        Returns:
            InnovationMetrics: 融合后的指标
        """
        # 将指标转换为矩阵进行融合
        metrics_matrix = np.array([[
            m.novelty, m.feasibility, m.impact, m.urgency,
            m.resource_requirement, m.risk_level, m.market_potential,
            m.technical_maturity, m.business_value, m.scalability
        ] for m in metrics_list])
        
        # 使用加权平均进行融合
        weights = np.array([1.0] * len(metrics_list))
        weights = weights / weights.sum()
        
        fused_values = np.average(metrics_matrix, axis=0, weights=weights)
        
        return InnovationMetrics(
            novelty=fused_values[0],
            feasibility=fused_values[1],
            impact=fused_values[2],
            urgency=fused_values[3],
            resource_requirement=fused_values[4],
            risk_level=fused_values[5],
            market_potential=fused_values[6],
            technical_maturity=fused_values[7],
            business_value=fused_values[8],
            scalability=fused_values[9]
        )
    
    def evaluate_innovation_state(self, innovation_id: str) -> Dict[str, Any]:
        """
        评估创新状态
        
        Args:
            innovation_id: 创新项目ID
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            innovation = self.get_innovation_state(innovation_id)
            if not innovation:
                return {"error": "未找到创新项目"}
            
            # 多维度评估
            evaluation = {
                "innovation_id": innovation_id,
                "evaluation_timestamp": datetime.now().isoformat(),
                "overall_score": innovation.innovation_score,
                "dimension_scores": {
                    "technical": self._evaluate_technical_dimension(innovation),
                    "business": self._evaluate_business_dimension(innovation),
                    "strategic": self._evaluate_strategic_dimension(innovation),
                    "operational": self._evaluate_operational_dimension(innovation)
                },
                "risk_assessment": self._assess_risks(innovation),
                "opportunity_analysis": self._analyze_opportunities(innovation),
                "recommendations": self._generate_evaluation_recommendations(innovation)
            }
            
            # 记录评估历史
            self._record_evaluation(innovation_id, evaluation)
            
            self.logger.info(f"完成创新项目 {innovation_id} 的状态评估")
            return evaluation
            
        except Exception as e:
            self.logger.error(f"创新状态评估失败: {e}")
            return {"error": str(e)}
    
    def _evaluate_technical_dimension(self, innovation: InnovationState) -> Dict[str, float]:
        """评估技术维度"""
        return {
            "technical_feasibility": innovation.metrics.feasibility,
            "maturity_level": innovation.metrics.technical_maturity,
            "complexity_score": 1.0 - innovation.metrics.feasibility,
            "innovation_potential": innovation.metrics.novelty
        }
    
    def _evaluate_business_dimension(self, innovation: InnovationState) -> Dict[str, float]:
        """评估商业维度"""
        return {
            "market_potential": innovation.metrics.market_potential,
            "business_value": innovation.metrics.business_value,
            "revenue_potential": innovation.metrics.impact,
            "cost_benefit_ratio": innovation.metrics.impact / max(innovation.metrics.resource_requirement, 0.1)
        }
    
    def _evaluate_strategic_dimension(self, innovation: InnovationState) -> Dict[str, float]:
        """评估战略维度"""
        return {
            "strategic_alignment": innovation.metrics.impact,
            "competitive_advantage": innovation.metrics.novelty,
            "long_term_value": innovation.metrics.business_value,
            "market_differentiation": innovation.metrics.novelty
        }
    
    def _evaluate_operational_dimension(self, innovation: InnovationState) -> Dict[str, float]:
        """评估运营维度"""
        return {
            "implementation_feasibility": innovation.metrics.feasibility,
            "resource_efficiency": 1.0 - innovation.metrics.resource_requirement,
            "time_to_market": 1.0 - innovation.metrics.urgency,
            "scalability_potential": innovation.metrics.scalability
        }
    
    def _assess_risks(self, innovation: InnovationState) -> Dict[str, Any]:
        """评估风险"""
        risk_factors = []
        risk_score = 0.0
        
        if innovation.metrics.risk_level > 0.7:
            risk_factors.append("高风险水平")
            risk_score += 30
        
        if innovation.metrics.feasibility < 0.3:
            risk_factors.append("技术可行性不足")
            risk_score += 25
        
        if innovation.metrics.resource_requirement > 0.8:
            risk_factors.append("资源需求过高")
            risk_score += 20
        
        if innovation.metrics.market_potential < 0.3:
            risk_factors.append("市场潜力有限")
            risk_score += 15
        
        if innovation.metrics.technical_maturity < 0.2:
            risk_factors.append("技术成熟度不足")
            risk_score += 10
        
        return {
            "risk_score": min(100, risk_score),
            "risk_level": "高" if risk_score > 70 else "中" if risk_score > 40 else "低",
            "risk_factors": risk_factors,
            "mitigation_strategies": self._generate_risk_mitigation_strategies(innovation)
        }
    
    def _analyze_opportunities(self, innovation: InnovationState) -> Dict[str, Any]:
        """分析机会"""
        opportunities = []
        opportunity_score = 0.0
        
        if innovation.metrics.market_potential > 0.7:
            opportunities.append("强大的市场潜力")
            opportunity_score += 25
        
        if innovation.metrics.novelty > 0.8:
            opportunities.append("高度创新性")
            opportunity_score += 25
        
        if innovation.metrics.impact > 0.7:
            opportunities.append("重大影响潜力")
            opportunity_score += 25
        
        if innovation.metrics.business_value > 0.6:
            opportunities.append("高商业价值")
            opportunity_score += 25
        
        return {
            "opportunity_score": min(100, opportunity_score),
            "opportunity_level": "高" if opportunity_score > 70 else "中" if opportunity_score > 40 else "低",
            "opportunities": opportunities,
            "leverage_strategies": self._generate_leverage_strategies(innovation)
        }
    
    def check_innovation_consistency(self, innovation_ids: List[str]) -> ConsistencyResult:
        """
        检验创新一致性
        
        Args:
            innovation_ids: 创新项目ID列表
            
        Returns:
            ConsistencyResult: 一致性检验结果
        """
        try:
            innovations = []
            for innovation_id in innovation_ids:
                innovation = self.get_innovation_state(innovation_id)
                if innovation:
                    innovations.append(innovation)
            
            conflicts = []
            synergies = []
            recommendations = []
            
            # 检查类型冲突
            type_conflicts = self._check_type_conflicts(innovations)
            conflicts.extend(type_conflicts)
            
            # 检查资源冲突
            resource_conflicts = self._check_resource_conflicts(innovations)
            conflicts.extend(resource_conflicts)
            
            # 检查目标冲突
            goal_conflicts = self._check_goal_conflicts(innovations)
            conflicts.extend(goal_conflicts)
            
            # 识别协同效应
            synergy_effects = self._identify_synergy_effects(innovations)
            synergies.extend(synergy_effects)
            
            # 生成建议
            recommendations = self._generate_consistency_recommendations(innovations, conflicts, synergies)
            
            # 计算一致性分数
            consistency_score = self._calculate_consistency_score(innovations, conflicts, synergies)
            
            is_consistent = len(conflicts) == 0 or len(conflicts) < len(synergies)
            
            return ConsistencyResult(
                is_consistent=is_consistent,
                conflicts=conflicts,
                synergies=synergies,
                recommendations=recommendations,
                consistency_score=consistency_score
            )
            
        except Exception as e:
            self.logger.error(f"一致性检验失败: {e}")
            return ConsistencyResult(
                is_consistent=False,
                conflicts=[f"检验过程出错: {e}"],
                synergies=[],
                recommendations=["请检查系统状态并重试"],
                consistency_score=0.0
            )
    
    def _check_type_conflicts(self, innovations: List[InnovationState]) -> List[str]:
        """检查类型冲突"""
        conflicts = []
        type_groups = defaultdict(list)
        
        # 按类型分组
        for innovation in innovations:
            type_groups[innovation.type].append(innovation)
        
        # 检查互斥类型组合
        for type_name, group in type_groups.items():
            if len(group) > 3:  # 同一类型项目过多
                conflicts.append(f"类型 '{type_name}' 项目过多 ({len(group)} 个)，可能存在资源竞争")
        
        return conflicts
    
    def _check_resource_conflicts(self, innovations: List[InnovationState]) -> List[str]:
        """检查资源冲突"""
        conflicts = []
        total_budget = sum(innovation.budget for innovation in innovations)
        total_progress = sum(innovation.progress for innovation in innovations)
        
        if total_budget > 1000000:  # 假设预算上限
            conflicts.append("项目总预算超出限制")
        
        if total_progress > len(innovations) * 100:  # 进度总和异常
            conflicts.append("项目进度数据异常")
        
        return conflicts
    
    def _check_goal_conflicts(self, innovations: List[InnovationState]) -> List[str]:
        """检查目标冲突"""
        conflicts = []
        
        # 检查标签冲突
        all_tags = []
        for innovation in innovations:
            all_tags.extend(innovation.tags)
        
        tag_counts = defaultdict(int)
        for tag in all_tags:
            tag_counts[tag] += 1
        
        # 相同标签过多可能表示目标重复
        for tag, count in tag_counts.items():
            if count > len(innovations) * 0.5:
                conflicts.append(f"标签 '{tag}' 在多个项目中重复，可能存在目标重叠")
        
        return conflicts
    
    def _identify_synergy_effects(self, innovations: List[InnovationState]) -> List[str]:
        """识别协同效应"""
        synergies = []
        
        # 技术协同
        technical_innovations = [i for i in innovations if i.type == InnovationType.TECHNICAL]
        if len(technical_innovations) > 1:
            synergies.append("多个技术创新项目可形成技术协同效应")
        
        # 市场协同
        market_innovations = [i for i in innovations if i.type == InnovationType.MARKET]
        if len(market_innovations) > 1:
            synergies.append("多个市场创新项目可形成市场协同效应")
        
        # 跨类型协同
        if len(set(i.type for i in innovations)) > 2:
            synergies.append("跨类型创新项目可形成多元化协同效应")
        
        return synergies
    
    def _calculate_consistency_score(self, innovations: List[InnovationState], 
                                   conflicts: List[str], synergies: List[str]) -> float:
        """计算一致性分数"""
        base_score = 100.0
        
        # 冲突扣分
        conflict_penalty = len(conflicts) * 10
        base_score -= conflict_penalty
        
        # 协同加分
        synergy_bonus = len(synergies) * 5
        base_score += synergy_bonus
        
        return max(0.0, min(100.0, base_score))
    
    def prioritize_innovations(self, innovation_ids: List[str], 
                             criteria: Dict[str, float] = None) -> List[PriorityResult]:
        """
        创新优先级排序
        
        Args:
            innovation_ids: 创新项目ID列表
            criteria: 排序标准权重
            
        Returns:
            List[PriorityResult]: 优先级排序结果
        """
        try:
            if criteria is None:
                criteria = {
                    "innovation_score": 0.3,
                    "feasibility": 0.2,
                    "impact": 0.2,
                    "urgency": 0.15,
                    "market_potential": 0.15
                }
            
            results = []
            
            for innovation_id in innovation_ids:
                innovation = self.get_innovation_state(innovation_id)
                if not innovation:
                    continue
                
                # 计算优先级分数
                priority_score = self._calculate_priority_score(innovation, criteria)
                
                # 确定优先级级别
                priority_level = self._determine_priority_level(priority_score)
                
                result = PriorityResult(
                    innovation_id=innovation_id,
                    priority_score=priority_score,
                    priority_level=priority_level,
                    ranking=0,  # 将在排序后设置
                    factors=self._analyze_priority_factors(innovation, criteria)
                )
                
                results.append(result)
            
            # 按优先级分数排序
            results.sort(key=lambda x: x.priority_score, reverse=True)
            
            # 设置排名
            for i, result in enumerate(results):
                result.ranking = i + 1
            
            self.logger.info(f"完成 {len(results)} 个创新项目的优先级排序")
            return results
            
        except Exception as e:
            self.logger.error(f"优先级排序失败: {e}")
            return []
    
    def _calculate_priority_score(self, innovation: InnovationState, 
                                criteria: Dict[str, float]) -> float:
        """计算优先级分数"""
        score = 0.0
        
        for factor, weight in criteria.items():
            if factor == "innovation_score":
                score += innovation.innovation_score * weight
            elif factor == "feasibility":
                score += innovation.metrics.feasibility * 100 * weight
            elif factor == "impact":
                score += innovation.metrics.impact * 100 * weight
            elif factor == "urgency":
                score += innovation.metrics.urgency * 100 * weight
            elif factor == "market_potential":
                score += innovation.metrics.market_potential * 100 * weight
        
        return score
    
    def _determine_priority_level(self, score: float) -> PriorityLevel:
        """确定优先级级别"""
        if score >= 80:
            return PriorityLevel.CRITICAL
        elif score >= 60:
            return PriorityLevel.HIGH
        elif score >= 40:
            return PriorityLevel.MEDIUM
        else:
            return PriorityLevel.LOW
    
    def _analyze_priority_factors(self, innovation: InnovationState, 
                                criteria: Dict[str, float]) -> Dict[str, float]:
        """分析优先级因素"""
        factors = {}
        
        for factor, weight in criteria.items():
            if factor == "innovation_score":
                factors[factor] = innovation.innovation_score
            elif factor == "feasibility":
                factors[factor] = innovation.metrics.feasibility * 100
            elif factor == "impact":
                factors[factor] = innovation.metrics.impact * 100
            elif factor == "urgency":
                factors[factor] = innovation.metrics.urgency * 100
            elif factor == "market_potential":
                factors[factor] = innovation.metrics.market_potential * 100
        
        return factors
    
    def get_historical_records(self, innovation_id: str = None, 
                             start_date: datetime = None, 
                             end_date: datetime = None) -> List[HistoricalRecord]:
        """
        获取创新状态历史记录
        
        Args:
            innovation_id: 创新项目ID（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        Returns:
            List[HistoricalRecord]: 历史记录列表
        """
        try:
            query = "SELECT * FROM historical_records WHERE 1=1"
            params = []
            
            if innovation_id:
                query += " AND innovation_id = ?"
                params.append(innovation_id)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC"
            
            records = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                for row in cursor.fetchall():
                    record = HistoricalRecord(
                        timestamp=datetime.fromisoformat(row[1]),
                        innovation_id=row[2],
                        state=json.loads(row[3]) if row[3] else {},
                        metrics=InnovationMetrics(**json.loads(row[4])) if row[4] else InnovationMetrics(),
                        events=json.loads(row[5]) if row[5] else [],
                        performance_indicators=json.loads(row[6]) if row[6] else {}
                    )
                    records.append(record)
            
            return records
            
        except Exception as e:
            self.logger.error(f"获取历史记录失败: {e}")
            return []
    
    def generate_innovation_report(self, innovation_ids: List[str] = None,
                                 report_type: str = "comprehensive") -> Dict[str, Any]:
        """
        生成创新状态报告
        
        Args:
            innovation_ids: 创新项目ID列表（可选）
            report_type: 报告类型 ("comprehensive", "summary", "trend")
            
        Returns:
            Dict[str, Any]: 报告内容
        """
        try:
            if innovation_ids is None:
                innovation_ids = list(self._innovation_cache.keys())
            
            # 获取所有创新项目
            innovations = []
            for innovation_id in innovation_ids:
                innovation = self.get_innovation_state(innovation_id)
                if innovation:
                    innovations.append(innovation)
            
            if not innovations:
                return {"error": "未找到创新项目"}
            
            # 生成报告
            if report_type == "comprehensive":
                return self._generate_comprehensive_report(innovations)
            elif report_type == "summary":
                return self._generate_summary_report(innovations)
            elif report_type == "trend":
                return self._generate_trend_report(innovations)
            else:
                return {"error": f"不支持的报告类型: {report_type}"}
                
        except Exception as e:
            self.logger.error(f"生成创新报告失败: {e}")
            return {"error": str(e)}
    
    def _generate_comprehensive_report(self, innovations: List[InnovationState]) -> Dict[str, Any]:
        """生成综合报告"""
        report = {
            "report_type": "comprehensive",
            "generation_time": datetime.now().isoformat(),
            "summary": {
                "total_innovations": len(innovations),
                "active_innovations": len([i for i in innovations if i.status != InnovationStatus.MAINTENANCE]),
                "average_innovation_score": np.mean([i.innovation_score for i in innovations]),
                "total_budget": sum(i.budget for i in innovations),
                "average_progress": np.mean([i.progress for i in innovations])
            },
            "status_distribution": self._analyze_status_distribution(innovations),
            "type_distribution": self._analyze_type_distribution(innovations),
            "priority_analysis": self._analyze_priority_distribution(innovations),
            "performance_metrics": self._calculate_performance_metrics(innovations),
            "risk_analysis": self._analyze_overall_risks(innovations),
            "recommendations": self._generate_comprehensive_recommendations(innovations)
        }
        
        return report
    
    def _generate_summary_report(self, innovations: List[InnovationState]) -> Dict[str, Any]:
        """生成摘要报告"""
        return {
            "report_type": "summary",
            "generation_time": datetime.now().isoformat(),
            "key_metrics": {
                "total_count": len(innovations),
                "high_priority_count": len([i for i in innovations if i.priority == PriorityLevel.HIGH]),
                "critical_priority_count": len([i for i in innovations if i.priority == PriorityLevel.CRITICAL]),
                "average_score": np.mean([i.innovation_score for i in innovations]) if innovations else 0
            },
            "top_performers": self._get_top_performing_innovations(innovations, 5),
            "critical_issues": self._identify_critical_issues(innovations)
        }
    
    def _generate_trend_report(self, innovations: List[InnovationState]) -> Dict[str, Any]:
        """生成趋势报告"""
        # 获取历史数据
        historical_records = []
        for innovation in innovations:
            records = self.get_historical_records(innovation.id)
            historical_records.extend(records)
        
        # 分析趋势
        trends = self._analyze_innovation_trends(historical_records)
        
        return {
            "report_type": "trend",
            "generation_time": datetime.now().isoformat(),
            "trends": trends,
            "predictions": self._predict_future_trends(historical_records),
            "recommendations": self._generate_trend_recommendations(trends)
        }
    
    def check_innovation_alerts(self) -> List[Dict[str, Any]]:
        """
        检查创新状态预警
        
        Returns:
            List[Dict[str, Any]]: 预警列表
        """
        alerts = []
        
        try:
            # 检查各种预警条件
            for innovation_id, innovation in self._innovation_cache.items():
                # 进度预警
                progress_alerts = self._check_progress_alerts(innovation)
                alerts.extend(progress_alerts)
                
                # 风险预警
                risk_alerts = self._check_risk_alerts(innovation)
                alerts.extend(risk_alerts)
                
                # 时间预警
                time_alerts = self._check_time_alerts(innovation)
                alerts.extend(time_alerts)
                
                # 资源预警
                resource_alerts = self._check_resource_alerts(innovation)
                alerts.extend(resource_alerts)
            
            # 保存预警记录
            self._save_alert_records(alerts)
            
            self.logger.info(f"检测到 {len(alerts)} 个预警")
            return alerts
            
        except Exception as e:
            self.logger.error(f"检查预警失败: {e}")
            return []
    
    def _check_progress_alerts(self, innovation: InnovationState) -> List[Dict[str, Any]]:
        """检查进度预警"""
        alerts = []
        
        # 进度滞后预警
        if innovation.progress < 30 and innovation.status in [InnovationStatus.DEVELOPMENT, InnovationStatus.TESTING]:
            alerts.append({
                "innovation_id": innovation.id,
                "alert_type": "进度滞后",
                "severity": "high",
                "message": f"创新项目 {innovation.name} 进度滞后，当前进度 {innovation.progress}%",
                "timestamp": datetime.now().isoformat()
            })
        
        # 长期无进展预警
        days_since_update = (datetime.now() - innovation.updated_at).days
        if days_since_update > 30 and innovation.status not in [InnovationStatus.MAINTENANCE, InnovationStatus.DEPLOYMENT]:
            alerts.append({
                "innovation_id": innovation.id,
                "alert_type": "长期无进展",
                "severity": "medium",
                "message": f"创新项目 {innovation.name} 已 {days_since_update} 天无更新",
                "timestamp": datetime.now().isoformat()
            })
        
        return alerts
    
    def _check_risk_alerts(self, innovation: InnovationState) -> List[Dict[str, Any]]:
        """检查风险预警"""
        alerts = []
        
        if innovation.metrics.risk_level > 0.8:
            alerts.append({
                "innovation_id": innovation.id,
                "alert_type": "高风险",
                "severity": "critical",
                "message": f"创新项目 {innovation.name} 风险水平过高 ({innovation.metrics.risk_level:.2f})",
                "timestamp": datetime.now().isoformat()
            })
        
        return alerts
    
    def _check_time_alerts(self, innovation: InnovationState) -> List[Dict[str, Any]]:
        """检查时间预警"""
        alerts = []
        
        if innovation.timeline and "deadline" in innovation.timeline:
            deadline = innovation.timeline["deadline"]
            days_until_deadline = (deadline - datetime.now()).days
            
            if days_until_deadline < 0:
                alerts.append({
                    "innovation_id": innovation.id,
                    "alert_type": "已逾期",
                    "severity": "critical",
                    "message": f"创新项目 {innovation.name} 已逾期 {abs(days_until_deadline)} 天",
                    "timestamp": datetime.now().isoformat()
                })
            elif days_until_deadline < 7:
                alerts.append({
                    "innovation_id": innovation.id,
                    "alert_type": "即将逾期",
                    "severity": "high",
                    "message": f"创新项目 {innovation.name} 将在 {days_until_deadline} 天后逾期",
                    "timestamp": datetime.now().isoformat()
                })
        
        return alerts
    
    def _check_resource_alerts(self, innovation: InnovationState) -> List[Dict[str, Any]]:
        """检查资源预警"""
        alerts = []
        
        if innovation.metrics.resource_requirement > 0.9:
            alerts.append({
                "innovation_id": innovation.id,
                "alert_type": "资源需求过高",
                "severity": "medium",
                "message": f"创新项目 {innovation.name} 资源需求过高",
                "timestamp": datetime.now().isoformat()
            })
        
        return alerts
    
    def get_innovation_state(self, innovation_id: str) -> Optional[InnovationState]:
        """
        获取创新状态
        
        Args:
            innovation_id: 创新项目ID
            
        Returns:
            Optional[InnovationState]: 创新状态对象
        """
        # 先从缓存获取
        if innovation_id in self._innovation_cache:
            return self._innovation_cache[innovation_id]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM innovation_states WHERE id = ?', (innovation_id,))
                row = cursor.fetchone()
                
                if row:
                    innovation = InnovationState(
                        id=row[0],
                        name=row[1],
                        type=InnovationType(row[2]),
                        status=InnovationStatus(row[3]),
                        description=row[4] or "",
                        metrics=InnovationMetrics(**json.loads(row[5])) if row[5] else InnovationMetrics(),
                        priority=PriorityLevel(row[6]),
                        created_at=datetime.fromisoformat(row[7]),
                        updated_at=datetime.fromisoformat(row[8]),
                        owner=row[9] or "",
                        tags=json.loads(row[10]) if row[10] else [],
                        dependencies=json.loads(row[11]) if row[11] else [],
                        progress=row[12],
                        budget=row[13],
                        timeline=json.loads(row[14]) if row[14] else None,
                        stakeholders=json.loads(row[15]) if row[15] else None,
                        success_criteria=json.loads(row[16]) if row[16] else None,
                        risk_factors=json.loads(row[17]) if row[17] else None,
                        innovation_score=row[18]
                    )
                    
                    # 缓存到内存
                    self._innovation_cache[innovation_id] = innovation
                    return innovation
                
        except Exception as e:
            self.logger.error(f"获取创新状态失败: {e}")
        
        return None
    
    def update_innovation_state(self, innovation_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新创新状态
        
        Args:
            innovation_id: 创新项目ID
            updates: 更新内容
            
        Returns:
            bool: 更新是否成功
        """
        try:
            innovation = self.get_innovation_state(innovation_id)
            if not innovation:
                return False
            
            # 更新字段
            for key, value in updates.items():
                if hasattr(innovation, key):
                    setattr(innovation, key, value)
            
            # 更新时间
            innovation.updated_at = datetime.now()
            
            # 重新计算创新分数
            if 'metrics' in updates:
                innovation.innovation_score = self._calculate_innovation_score(innovation.metrics)
            
            # 保存到数据库
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE innovation_states 
                    SET name = ?, type = ?, status = ?, description = ?, metrics = ?, 
                        priority = ?, updated_at = ?, owner = ?, tags = ?, dependencies = ?, 
                        progress = ?, budget = ?, timeline = ?, stakeholders = ?, 
                        success_criteria = ?, risk_factors = ?, innovation_score = ?
                    WHERE id = ?
                ''', (
                    innovation.name, innovation.type.value, innovation.status.value,
                    innovation.description, json.dumps(asdict(innovation.metrics), cls=InnovationJSONEncoder),
                    innovation.priority.value, innovation.updated_at.isoformat(),
                    innovation.owner, json.dumps(innovation.tags),
                    json.dumps(innovation.dependencies), innovation.progress,
                    innovation.budget, json.dumps(innovation.timeline) if innovation.timeline else None,
                    json.dumps(innovation.stakeholders) if innovation.stakeholders else None,
                    json.dumps(innovation.success_criteria) if innovation.success_criteria else None,
                    json.dumps(innovation.risk_factors) if innovation.risk_factors else None,
                    innovation.innovation_score, innovation.id
                ))
                conn.commit()
            
            # 更新缓存
            self._innovation_cache[innovation_id] = innovation
            
            # 记录历史
            self._record_historical_event(innovation_id, "状态更新", innovation)
            
            self.logger.info(f"创新项目 {innovation_id} 状态更新成功")
            return True
            
        except Exception as e:
            self.logger.error(f"更新创新状态失败: {e}")
            return False
    
    def _record_historical_event(self, innovation_id: str, event: str, innovation: InnovationState):
        """记录历史事件"""
        try:
            record = HistoricalRecord(
                timestamp=datetime.now(),
                innovation_id=innovation_id,
                state=asdict(innovation),
                metrics=innovation.metrics,
                events=[event],
                performance_indicators={
                    "innovation_score": innovation.innovation_score,
                    "progress": innovation.progress,
                    "risk_level": innovation.metrics.risk_level
                }
            )
            
            # 保存到数据库
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO historical_records 
                    (timestamp, innovation_id, state, metrics, events, performance_indicators)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    record.timestamp.isoformat(),
                    record.innovation_id,
                    json.dumps(record.state, cls=InnovationJSONEncoder),
                    json.dumps(asdict(record.metrics), cls=InnovationJSONEncoder),
                    json.dumps(record.events),
                    json.dumps(record.performance_indicators)
                ))
                conn.commit()
            
            # 添加到内存队列
            self._historical_data.append(record)
            
        except Exception as e:
            self.logger.error(f"记录历史事件失败: {e}")
    
    def _record_evaluation(self, innovation_id: str, evaluation: Dict[str, Any]):
        """记录评估历史"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO historical_records 
                    (timestamp, innovation_id, state, metrics, events, performance_indicators)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    innovation_id,
                    json.dumps({"evaluation": evaluation}),
                    json.dumps({}),
                    json.dumps(["状态评估"]),
                    json.dumps({
                        "overall_score": evaluation.get("overall_score", 0),
                        "risk_score": evaluation.get("risk_assessment", {}).get("risk_score", 0),
                        "opportunity_score": evaluation.get("opportunity_analysis", {}).get("opportunity_score", 0)
                    })
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"记录评估历史失败: {e}")
    
    def _save_alert_records(self, alerts: List[Dict[str, Any]]):
        """保存预警记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for alert in alerts:
                    cursor.execute('''
                        INSERT INTO alert_records 
                        (timestamp, innovation_id, alert_type, severity, message)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        alert["timestamp"],
                        alert["innovation_id"],
                        alert["alert_type"],
                        alert["severity"],
                        alert["message"]
                    ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"保存预警记录失败: {e}")
    
    # 辅助方法实现
    def _generate_fusion_recommendations(self, innovations: List[InnovationState], 
                                       consistency: ConsistencyResult) -> List[str]:
        """生成融合建议"""
        recommendations = []
        
        if not consistency.is_consistent:
            recommendations.append("建议解决冲突后再进行融合")
        
        if len(innovations) > 5:
            recommendations.append("建议分批进行融合，避免复杂度过高")
        
        if consistency.synergies:
            recommendations.append("充分利用识别的协同效应")
        
        return recommendations
    
    def _calculate_synergies(self, innovations: List[InnovationState]) -> Dict[str, float]:
        """计算协同效应"""
        synergies = {}
        
        # 技术协同
        tech_count = len([i for i in innovations if i.type == InnovationType.TECHNICAL])
        synergies["technical_synergy"] = min(1.0, tech_count * 0.3)
        
        # 市场协同
        market_count = len([i for i in innovations if i.type == InnovationType.MARKET])
        synergies["market_synergy"] = min(1.0, market_count * 0.3)
        
        # 资源协同
        avg_resource = np.mean([i.metrics.resource_requirement for i in innovations])
        synergies["resource_synergy"] = 1.0 - avg_resource
        
        return synergies
    
    def _generate_evaluation_recommendations(self, innovation: InnovationState) -> List[str]:
        """生成评估建议"""
        recommendations = []
        
        if innovation.metrics.feasibility < 0.5:
            recommendations.append("提高技术可行性")
        
        if innovation.metrics.impact < 0.5:
            recommendations.append("增强项目影响力")
        
        if innovation.metrics.risk_level > 0.7:
            recommendations.append("降低项目风险")
        
        if innovation.progress < 50:
            recommendations.append("加快项目进度")
        
        return recommendations
    
    def _generate_risk_mitigation_strategies(self, innovation: InnovationState) -> List[str]:
        """生成风险缓解策略"""
        strategies = []
        
        if innovation.metrics.risk_level > 0.7:
            strategies.append("制定详细的风险应对计划")
            strategies.append("增加项目监控频率")
        
        if innovation.metrics.feasibility < 0.5:
            strategies.append("进行可行性研究")
            strategies.append("寻求技术专家支持")
        
        if innovation.metrics.resource_requirement > 0.8:
            strategies.append("优化资源配置")
            strategies.append("寻求额外资金支持")
        
        return strategies
    
    def _generate_leverage_strategies(self, innovation: InnovationState) -> List[str]:
        """生成杠杆策略"""
        strategies = []
        
        if innovation.metrics.market_potential > 0.7:
            strategies.append("加大市场推广投入")
        
        if innovation.metrics.novelty > 0.8:
            strategies.append("申请知识产权保护")
        
        if innovation.metrics.impact > 0.7:
            strategies.append("扩大项目规模")
        
        return strategies
    
    def _generate_consistency_recommendations(self, innovations: List[InnovationState],
                                            conflicts: List[str], synergies: List[str]) -> List[str]:
        """生成一致性建议"""
        recommendations = []
        
        if conflicts:
            recommendations.append("优先解决识别的冲突")
        
        if synergies:
            recommendations.append("积极利用协同效应")
        
        if len(innovations) > 10:
            recommendations.append("考虑项目组合优化")
        
        return recommendations
    
    def _generate_comprehensive_recommendations(self, innovations: List[InnovationState]) -> List[str]:
        """生成综合建议"""
        recommendations = []
        
        avg_score = np.mean([i.innovation_score for i in innovations])
        if avg_score < 50:
            recommendations.append("整体创新分数偏低，建议重新评估项目组合")
        
        high_risk_count = len([i for i in innovations if i.metrics.risk_level > 0.7])
        if high_risk_count > len(innovations) * 0.3:
            recommendations.append("高风险项目比例过高，建议加强风险管理")
        
        low_progress_count = len([i for i in innovations if i.progress < 30])
        if low_progress_count > len(innovations) * 0.5:
            recommendations.append("多个项目进度滞后，建议重新分配资源")
        
        return recommendations
    
    def _generate_trend_recommendations(self, trends: Dict[str, Any]) -> List[str]:
        """生成趋势建议"""
        recommendations = []
        
        # 基于趋势分析生成建议
        if trends.get("declining_performance", 0) > 0:
            recommendations.append("性能下降趋势需要关注")
        
        if trends.get("increasing_risks", 0) > 0:
            recommendations.append("风险上升趋势需要应对")
        
        return recommendations
    
    def _analyze_status_distribution(self, innovations: List[InnovationState]) -> Dict[str, int]:
        """分析状态分布"""
        distribution = defaultdict(int)
        for innovation in innovations:
            distribution[innovation.status.value] += 1
        return dict(distribution)
    
    def _analyze_type_distribution(self, innovations: List[InnovationState]) -> Dict[str, int]:
        """分析类型分布"""
        distribution = defaultdict(int)
        for innovation in innovations:
            distribution[innovation.type.value] += 1
        return dict(distribution)
    
    def _analyze_priority_distribution(self, innovations: List[InnovationState]) -> Dict[str, int]:
        """分析优先级分布"""
        distribution = defaultdict(int)
        for innovation in innovations:
            distribution[innovation.priority.value] += 1
        return dict(distribution)
    
    def _calculate_performance_metrics(self, innovations: List[InnovationState]) -> Dict[str, float]:
        """计算性能指标"""
        return {
            "average_innovation_score": np.mean([i.innovation_score for i in innovations]),
            "average_progress": np.mean([i.progress for i in innovations]),
            "average_risk_level": np.mean([i.metrics.risk_level for i in innovations]),
            "total_budget": sum(i.budget for i in innovations),
            "success_rate": len([i for i in innovations if i.progress > 80]) / len(innovations) * 100
        }
    
    def _analyze_overall_risks(self, innovations: List[InnovationState]) -> Dict[str, Any]:
        """分析整体风险"""
        risk_levels = [i.metrics.risk_level for i in innovations]
        return {
            "average_risk": np.mean(risk_levels),
            "high_risk_count": len([r for r in risk_levels if r > 0.7]),
            "risk_distribution": {
                "low": len([r for r in risk_levels if r < 0.3]),
                "medium": len([r for r in risk_levels if 0.3 <= r <= 0.7]),
                "high": len([r for r in risk_levels if r > 0.7])
            }
        }
    
    def _get_top_performing_innovations(self, innovations: List[InnovationState], count: int) -> List[Dict[str, Any]]:
        """获取表现最佳的创新项目"""
        sorted_innovations = sorted(innovations, key=lambda x: x.innovation_score, reverse=True)
        return [
            {
                "id": i.id,
                "name": i.name,
                "score": i.innovation_score,
                "progress": i.progress
            }
            for i in sorted_innovations[:count]
        ]
    
    def _identify_critical_issues(self, innovations: List[InnovationState]) -> List[Dict[str, Any]]:
        """识别关键问题"""
        issues = []
        
        for innovation in innovations:
            if innovation.metrics.risk_level > 0.8:
                issues.append({
                    "innovation_id": innovation.id,
                    "issue": "高风险",
                    "description": f"项目 {innovation.name} 风险水平过高"
                })
            
            if innovation.progress < 20 and innovation.status != InnovationStatus.CONCEPT:
                issues.append({
                    "innovation_id": innovation.id,
                    "issue": "进度滞后",
                    "description": f"项目 {innovation.name} 进度严重滞后"
                })
        
        return issues
    
    def _analyze_innovation_trends(self, historical_records: List[HistoricalRecord]) -> Dict[str, Any]:
        """分析创新趋势"""
        if not historical_records:
            return {}
        
        # 按时间排序
        sorted_records = sorted(historical_records, key=lambda x: x.timestamp)
        
        # 分析分数趋势
        scores = [r.performance_indicators.get("innovation_score", 0) for r in sorted_records]
        progress_values = [r.performance_indicators.get("progress", 0) for r in sorted_records]
        
        return {
            "score_trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "declining",
            "progress_trend": "improving" if len(progress_values) > 1 and progress_values[-1] > progress_values[0] else "declining",
            "record_count": len(historical_records),
            "time_span_days": (sorted_records[-1].timestamp - sorted_records[0].timestamp).days if len(sorted_records) > 1 else 0
        }
    
    def _predict_future_trends(self, historical_records: List[HistoricalRecord]) -> Dict[str, Any]:
        """预测未来趋势"""
        # 简单的线性趋势预测
        if len(historical_records) < 2:
            return {"prediction": "数据不足，无法预测"}
        
        # 这里可以添加更复杂的预测算法
        return {
            "prediction": "基于历史数据的趋势预测",
            "confidence": "medium",
            "recommended_actions": ["持续监控", "定期评估"]
        }
    
    def export_innovation_data(self, innovation_ids: List[str] = None, 
                             format: str = "json") -> str:
        """
        导出创新数据
        
        Args:
            innovation_ids: 创新项目ID列表
            format: 导出格式 ("json", "csv")
            
        Returns:
            str: 导出的数据
        """
        try:
            if innovation_ids is None:
                innovation_ids = list(self._innovation_cache.keys())
            
            innovations = []
            for innovation_id in innovation_ids:
                innovation = self.get_innovation_state(innovation_id)
                if innovation:
                    innovations.append(asdict(innovation))
            
            if format == "json":
                return json.dumps(innovations, ensure_ascii=False, indent=2, default=str)
            elif format == "csv":
                # 简化的CSV导出
                import csv
                import io
                
                output = io.StringIO()
                if innovations:
                    fieldnames = innovations[0].keys()
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    for innovation in innovations:
                        # 简化复杂字段
                        simple_innovation = {}
                        for key, value in innovation.items():
                            if isinstance(value, (dict, list)):
                                simple_innovation[key] = str(value)
                            else:
                                simple_innovation[key] = value
                        writer.writerow(simple_innovation)
                
                return output.getvalue()
            else:
                return json.dumps({"error": f"不支持的导出格式: {format}"})
                
        except Exception as e:
            self.logger.error(f"导出数据失败: {e}")
            return json.dumps({"error": str(e)})


# 使用示例和测试代码
def demo_innovation_aggregator():
    """演示创新状态聚合器的使用"""
    
    # 创建聚合器实例
    aggregator = InnovationStateAggregator()
    
    # 创建示例创新项目
    innovation1 = InnovationState(
        id="INNO001",
        name="AI驱动的智能推荐系统",
        type=InnovationType.TECHNICAL,
        status=InnovationStatus.DEVELOPMENT,
        description="基于深度学习的个性化推荐系统",
        metrics=InnovationMetrics(
            novelty=0.9,
            feasibility=0.7,
            impact=0.8,
            urgency=0.6,
            resource_requirement=0.6,
            risk_level=0.4,
            market_potential=0.9,
            technical_maturity=0.6,
            business_value=0.8,
            scalability=0.9
        ),
        priority=PriorityLevel.HIGH,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        owner="张三",
        tags=["AI", "推荐", "机器学习"],
        dependencies=["DATA001", "INFRA001"],
        progress=45.0,
        budget=500000.0
    )
    
    innovation2 = InnovationState(
        id="INNO002",
        name="区块链供应链管理平台",
        type=InnovationType.PRODUCT,
        status=InnovationStatus.TESTING,
        description="基于区块链的透明供应链管理系统",
        metrics=InnovationMetrics(
            novelty=0.8,
            feasibility=0.6,
            impact=0.9,
            urgency=0.7,
            resource_requirement=0.8,
            risk_level=0.6,
            market_potential=0.8,
            technical_maturity=0.5,
            business_value=0.9,
            scalability=0.7
        ),
        priority=PriorityLevel.CRITICAL,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        owner="李四",
        tags=["区块链", "供应链", "透明度"],
        dependencies=["BLOCK001"],
        progress=70.0,
        budget=800000.0
    )
    
    # 注册创新项目
    aggregator.register_innovation(innovation1)
    aggregator.register_innovation(innovation2)
    
    # 融合创新状态
    fusion_result = aggregator.fuse_innovation_states(["INNO001", "INNO002"])
    print("融合结果:", json.dumps(fusion_result, ensure_ascii=False, indent=2))
    
    # 评估创新状态
    evaluation = aggregator.evaluate_innovation_state("INNO001")
    print("评估结果:", json.dumps(evaluation, ensure_ascii=False, indent=2))
    
    # 检查一致性
    consistency = aggregator.check_innovation_consistency(["INNO001", "INNO002"])
    print("一致性检验:", json.dumps(asdict(consistency), ensure_ascii=False, indent=2))
    
    # 优先级排序
    priorities = aggregator.prioritize_innovations(["INNO001", "INNO002"])
    for priority in priorities:
        print(f"优先级排序: {priority.innovation_id} - {priority.priority_score:.2f}")
    
    # 生成报告
    report = aggregator.generate_innovation_report(["INNO001", "INNO002"])
    print("创新报告:", json.dumps(report, ensure_ascii=False, indent=2))
    
    # 检查预警
    alerts = aggregator.check_innovation_alerts()
    print("预警信息:", json.dumps(alerts, ensure_ascii=False, indent=2))
    
    # 导出数据
    export_data = aggregator.export_innovation_data(["INNO001", "INNO002"])
    print("导出数据长度:", len(export_data))


if __name__ == "__main__":
    demo_innovation_aggregator()