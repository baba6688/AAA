#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F6层次学习管理器
实现多层次学习的协调、管理、优化和监控功能

Author: AI Assistant
Date: 2025-11-05
"""

import threading
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import uuid


class LearningLevel(Enum):
    """学习层次枚举"""
    RAW_DATA = "raw_data"           # 原始数据层
    FEATURE = "feature"             # 特征层
    PATTERN = "pattern"             # 模式层
    KNOWLEDGE = "knowledge"         # 知识层
    STRATEGY = "strategy"           # 策略层
    META = "meta"                   # 元认知层


class LearningState(Enum):
    """学习状态枚举"""
    IDLE = "idle"
    LEARNING = "learning"
    OPTIMIZING = "optimizing"
    EVALUATING = "evaluating"
    CONFLICTING = "conflicting"
    SYNCHRONIZING = "synchronizing"


class ConflictType(Enum):
    """冲突类型枚举"""
    RESOURCE_CONFLICT = "resource_conflict"
    GOAL_CONFLICT = "goal_conflict"
    STRATEGY_CONFLICT = "strategy_conflict"
    PRIORITY_CONFLICT = "priority_conflict"
    TIMING_CONFLICT = "timing_conflict"


@dataclass
class LearningNode:
    """学习节点"""
    id: str
    level: LearningLevel
    state: LearningState
    data: Dict[str, Any] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def update_performance(self, metrics: Dict[str, float]):
        """更新性能指标"""
        self.performance_metrics.update(metrics)
        self.timestamp = time.time()


@dataclass
class LearningTransfer:
    """学习传递对象"""
    source_level: LearningLevel
    target_level: LearningLevel
    data: Any
    transfer_type: str  # "knowledge", "pattern", "strategy"
    quality_score: float
    timestamp: float = field(default_factory=time.time)
    transfer_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ConflictInfo:
    """冲突信息"""
    conflict_id: str
    conflict_type: ConflictType
    involved_nodes: List[str]
    description: str
    severity: int  # 1-10
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False


class HierarchicalLearningManager:
    """F6层次学习管理器"""
    
    def __init__(self, max_levels: int = 6, enable_monitoring: bool = True):
        """
        初始化层次学习管理器
        
        Args:
            max_levels: 最大层次数
            enable_monitoring: 是否启用监控
        """
        self.max_levels = max_levels
        self.enable_monitoring = enable_monitoring
        
        # 核心组件
        self.learning_nodes: Dict[str, LearningNode] = {}
        self.learning_transfers: List[LearningTransfer] = []
        self.conflicts: Dict[str, ConflictInfo] = {}
        self.strategies: Dict[str, Any] = {}
        
        # 层次管理
        self.levels = list(LearningLevel)
        self.level_managers: Dict[LearningLevel, 'LevelManager'] = {}
        
        # 监控和评估
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 锁
        self.lock = threading.RLock()
        
        # 配置日志
        self._setup_logging()
        
        # 初始化各层次管理器
        self._initialize_level_managers()
        
        # 启动监控
        if self.enable_monitoring:
            self.start_monitoring()
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('HierarchicalLearningManager')
    
    def _initialize_level_managers(self):
        """初始化各层次管理器"""
        for level in self.levels:
            self.level_managers[level] = LevelManager(level, self)
    
    def create_learning_node(self, level: LearningLevel, node_id: Optional[str] = None) -> str:
        """
        创建学习节点
        
        Args:
            level: 学习层次
            node_id: 节点ID，如果为None则自动生成
            
        Returns:
            节点ID
        """
        if node_id is None:
            node_id = f"{level.value}_{uuid.uuid4().hex[:8]}"
        
        node = LearningNode(
            id=node_id,
            level=level,
            state=LearningState.IDLE
        )
        
        with self.lock:
            self.learning_nodes[node_id] = node
            self.logger.info(f"创建学习节点: {node_id} 在层次 {level.value}")
        
        return node_id
    
    def connect_nodes(self, source_id: str, target_id: str, bidirectional: bool = True):
        """
        连接学习节点
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            bidirectional: 是否双向连接
        """
        with self.lock:
            if source_id in self.learning_nodes and target_id in self.learning_nodes:
                self.learning_nodes[source_id].connections.append(target_id)
                if bidirectional:
                    self.learning_nodes[target_id].connections.append(source_id)
                self.logger.info(f"连接节点: {source_id} -> {target_id}")
    
    def transfer_learning(self, source_level: LearningLevel, target_level: LearningLevel, 
                         data: Any, transfer_type: str = "knowledge") -> str:
        """
        层次间学习传递
        
        Args:
            source_level: 源层次
            target_level: 目标层次
            data: 传递的数据
            transfer_type: 传递类型
            
        Returns:
            传递ID
        """
        # 评估传递质量
        quality_score = self._evaluate_transfer_quality(source_level, target_level, data)
        
        transfer = LearningTransfer(
            source_level=source_level,
            target_level=target_level,
            data=data,
            transfer_type=transfer_type,
            quality_score=quality_score
        )
        
        with self.lock:
            self.learning_transfers.append(transfer)
        
        # 触发目标层次的学习
        self._trigger_target_learning(target_level, transfer)
        
        self.logger.info(f"学习传递: {source_level.value} -> {target_level.value}, "
                        f"质量: {quality_score:.3f}")
        
        return transfer.transfer_id
    
    def _evaluate_transfer_quality(self, source_level: LearningLevel, target_level: LearningLevel, 
                                 data: Any) -> float:
        """评估传递质量"""
        # 基础质量评估
        base_quality = 0.7
        
        # 层次相关性评估
        level_distance = abs(self.levels.index(source_level) - self.levels.index(target_level))
        relevance_score = max(0.1, 1.0 - level_distance * 0.2)
        
        # 数据复杂度评估
        data_complexity = self._assess_data_complexity(data)
        complexity_score = min(1.0, data_complexity / 100.0)
        
        quality = base_quality * relevance_score * complexity_score
        return min(1.0, max(0.0, quality))
    
    def _assess_data_complexity(self, data: Any) -> float:
        """评估数据复杂度"""
        if isinstance(data, dict):
            return len(data) * 10 + sum(self._assess_data_complexity(v) for v in data.values())
        elif isinstance(data, (list, tuple)):
            return len(data) * 5 + sum(self._assess_data_complexity(item) for item in data)
        elif isinstance(data, str):
            return len(data) * 0.1
        else:
            return 1.0
    
    def _trigger_target_learning(self, target_level: LearningLevel, transfer: LearningTransfer):
        """触发目标层次的学习"""
        level_manager = self.level_managers[target_level]
        level_manager.process_transfer(transfer)
    
    def optimize_learning_hierarchy(self) -> Dict[str, Any]:
        """
        优化学习层次结构
        
        Returns:
            优化结果
        """
        self.logger.info("开始层次学习优化")
        
        optimization_results = {
            "timestamp": time.time(),
            "optimizations": [],
            "performance_improvements": {},
            "structure_changes": []
        }
        
        with self.lock:
            # 1. 评估当前层次性能
            level_performance = self._evaluate_level_performance()
            
            # 2. 识别需要优化的层次
            underperforming_level_keys = [
                level_key for level_key, perf in level_performance.items() 
                if perf['overall_score'] < 0.6
            ]
            
            # 3. 执行优化策略
            for level_key in underperforming_level_keys:
                # 转换回LearningLevel枚举
                level = next((l for l in self.levels if l.value == level_key), None)
                if level:
                    optimization = self._optimize_level(level, level_performance[level_key])
                optimization_results["optimizations"].append(optimization)
            
            # 4. 调整层次连接
            structure_changes = self._optimize_connections()
            optimization_results["structure_changes"] = structure_changes
            
            # 5. 更新性能指标
            new_performance = self._evaluate_level_performance()
            for level, perf in new_performance.items():
                if level in level_performance:
                    improvement = perf['overall_score'] - level_performance[level]['overall_score']
                    optimization_results["performance_improvements"][level] = improvement
        
        self.logger.info(f"层次优化完成，执行了 {len(optimization_results['optimizations'])} 项优化")
        return optimization_results
    
    def _evaluate_level_performance(self) -> Dict[str, Dict[str, float]]:
        """评估各层次性能"""
        performance = {}
        
        for level in self.levels:
            level_nodes = [
                node for node in self.learning_nodes.values() 
                if node.level == level
            ]
            
            if not level_nodes:
                performance[level.value] = {'overall_score': 0.0, 'efficiency': 0.0, 'quality': 0.0}
                continue
            
            # 计算效率指标
            efficiency_scores = []
            quality_scores = []
            
            for node in level_nodes:
                if node.performance_metrics:
                    efficiency_scores.append(node.performance_metrics.get('efficiency', 0.5))
                    quality_scores.append(node.performance_metrics.get('quality', 0.5))
            
            avg_efficiency = np.mean(efficiency_scores) if efficiency_scores else 0.5
            avg_quality = np.mean(quality_scores) if quality_scores else 0.5
            overall_score = (avg_efficiency + avg_quality) / 2
            
            performance[level.value] = {
                'overall_score': overall_score,
                'efficiency': avg_efficiency,
                'quality': avg_quality
            }
        
        return performance
    
    def _optimize_level(self, level: LearningLevel, performance: Dict[str, float]) -> Dict[str, Any]:
        """优化指定层次"""
        optimization = {
            "level": level.value,
            "actions": [],
            "expected_improvement": 0.0
        }
        
        level_manager = self.level_managers[level]
        
        if performance['efficiency'] < 0.5:
            # 优化效率
            actions = level_manager.optimize_efficiency()
            optimization["actions"].extend(actions)
            optimization["expected_improvement"] += 0.2
        
        if performance['quality'] < 0.5:
            # 优化质量
            actions = level_manager.optimize_quality()
            optimization["actions"].extend(actions)
            optimization["expected_improvement"] += 0.2
        
        self.logger.info(f"优化层次 {level.value}: {len(optimization['actions'])} 项措施")
        return optimization
    
    def _optimize_connections(self) -> List[Dict[str, Any]]:
        """优化节点连接"""
        changes = []
        
        # 识别低效连接
        for node_id, node in self.learning_nodes.items():
            if len(node.connections) > 10:  # 连接过多
                # 保留最有效的连接
                effective_connections = self._identify_effective_connections(node_id)
                removed_connections = set(node.connections) - set(effective_connections)
                
                for removed_id in removed_connections:
                    node.connections.remove(removed_id)
                    if removed_id in self.learning_nodes:
                        self.learning_nodes[removed_id].connections.remove(node_id)
                
                changes.append({
                    "type": "connection_pruning",
                    "node": node_id,
                    "removed_connections": list(removed_connections)
                })
        
        return changes
    
    def _identify_effective_connections(self, node_id: str) -> List[str]:
        """识别有效连接"""
        node = self.learning_nodes[node_id]
        # 简化的有效性评估：基于性能指标
        effective_connections = []
        
        for connected_id in node.connections:
            if connected_id in self.learning_nodes:
                connected_node = self.learning_nodes[connected_id]
                # 如果连接节点的性能较好，认为连接有效
                if connected_node.performance_metrics.get('quality', 0.5) > 0.6:
                    effective_connections.append(connected_id)
        
        return effective_connections[:5]  # 最多保留5个有效连接
    
    def evaluate_cross_level_effectiveness(self) -> Dict[str, Any]:
        """
        评估跨层次学习效果
        
        Returns:
            评估结果
        """
        self.logger.info("开始跨层次效果评估")
        
        evaluation_results = {
            "timestamp": time.time(),
            "cross_level_metrics": {},
            "transfer_effectiveness": {},
            "coordination_score": 0.0,
            "recommendations": []
        }
        
        with self.lock:
            # 1. 评估跨层次传递效果
            transfer_effectiveness = self._analyze_transfer_effectiveness()
            evaluation_results["transfer_effectiveness"] = transfer_effectiveness
            
            # 2. 评估层次协调性
            coordination_score = self._evaluate_coordination()
            evaluation_results["coordination_score"] = coordination_score
            
            # 3. 生成改进建议
            recommendations = self._generate_improvement_recommendations(transfer_effectiveness, coordination_score)
            evaluation_results["recommendations"] = recommendations
        
        return evaluation_results
    
    def _analyze_transfer_effectiveness(self) -> Dict[str, float]:
        """分析传递效果"""
        effectiveness = {}
        
        # 按层次对分析传递效果
        for i, source_level in enumerate(self.levels):
            for j, target_level in enumerate(self.levels):
                if i != j:
                    key = f"{source_level.value}_to_{target_level.value}"
                    
                    # 获取该方向的所有传递
                    relevant_transfers = [
                        t for t in self.learning_transfers
                        if t.source_level == source_level and t.target_level == target_level
                    ]
                    
                    if relevant_transfers:
                        avg_quality = np.mean([t.quality_score for t in relevant_transfers])
                        effectiveness[key] = avg_quality
                    else:
                        effectiveness[key] = 0.0
        
        return effectiveness
    
    def _evaluate_coordination(self) -> float:
        """评估层次协调性"""
        if not self.learning_nodes:
            return 0.0
        
        # 计算协调性指标
        coordination_scores = []
        
        for node_id, node in self.learning_nodes.items():
            # 活跃连接比例
            active_connections = len(node.connections)
            total_possible = len(self.learning_nodes) - 1
            connection_ratio = active_connections / max(1, total_possible)
            
            # 状态一致性
            state_consistency = 1.0 if node.state != LearningState.CONFLICTING else 0.0
            
            # 综合协调分数
            coordination_score = (connection_ratio + state_consistency) / 2
            coordination_scores.append(coordination_score)
        
        return np.mean(coordination_scores)
    
    def _generate_improvement_recommendations(self, transfer_effectiveness: Dict[str, float], 
                                            coordination_score: float) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于传递效果的建议
        low_effectiveness_transfers = [
            key for key, score in transfer_effectiveness.items() if score < 0.5
        ]
        
        if low_effectiveness_transfers:
            recommendations.append(f"改进 {len(low_effectiveness_transfers)} 个低效传递路径")
        
        # 基于协调性的建议
        if coordination_score < 0.6:
            recommendations.append("增强层次间协调机制")
        
        # 基于冲突的建议
        active_conflicts = [c for c in self.conflicts.values() if not c.resolved]
        if active_conflicts:
            recommendations.append(f"解决 {len(active_conflicts)} 个活跃冲突")
        
        return recommendations
    
    def detect_and_resolve_conflicts(self) -> Dict[str, Any]:
        """
        检测和解决学习层次冲突
        
        Returns:
            冲突解决结果
        """
        self.logger.info("开始冲突检测和解决")
        
        resolution_results = {
            "timestamp": time.time(),
            "detected_conflicts": [],
            "resolved_conflicts": [],
            "unresolved_conflicts": [],
            "resolution_strategies": []
        }
        
        with self.lock:
            # 1. 检测冲突
            detected_conflicts = self._detect_conflicts()
            resolution_results["detected_conflicts"] = [
                {"id": c.conflict_id, "type": c.conflict_type.value, "severity": c.severity}
                for c in detected_conflicts
            ]
            
            # 2. 解决冲突
            for conflict in detected_conflicts:
                resolution = self._resolve_conflict(conflict)
                
                if resolution["resolved"]:
                    resolution_results["resolved_conflicts"].append(conflict.conflict_id)
                else:
                    resolution_results["unresolved_conflicts"].append(conflict.conflict_id)
                
                resolution_results["resolution_strategies"].append(resolution)
        
        self.logger.info(f"冲突处理完成: {len(resolution_results['resolved_conflicts'])} 解决, "
                        f"{len(resolution_results['unresolved_conflicts'])} 未解决")
        
        return resolution_results
    
    def _detect_conflicts(self) -> List[ConflictInfo]:
        """检测冲突"""
        conflicts = []
        
        # 检测资源冲突
        resource_conflicts = self._detect_resource_conflicts()
        conflicts.extend(resource_conflicts)
        
        # 检测目标冲突
        goal_conflicts = self._detect_goal_conflicts()
        conflicts.extend(goal_conflicts)
        
        # 检测策略冲突
        strategy_conflicts = self._detect_strategy_conflicts()
        conflicts.extend(strategy_conflicts)
        
        # 更新冲突列表
        for conflict in conflicts:
            self.conflicts[conflict.conflict_id] = conflict
        
        return conflicts
    
    def _detect_resource_conflicts(self) -> List[ConflictInfo]:
        """检测资源冲突"""
        conflicts = []
        
        # 统计各节点资源使用
        resource_usage = defaultdict(int)
        for node in self.learning_nodes.values():
            resource_usage[node.level] += 1
        
        # 识别资源过载
        for level, usage in resource_usage.items():
            if usage > 10:  # 阈值可调
                involved_nodes = [
                    node.id for node in self.learning_nodes.values() 
                    if node.level == level
                ]
                
                conflict = ConflictInfo(
                    conflict_id=str(uuid.uuid4()),
                    conflict_type=ConflictType.RESOURCE_CONFLICT,
                    involved_nodes=involved_nodes,
                    description=f"层次 {level.value} 资源过载，使用节点数: {usage}",
                    severity=min(10, usage - 8)
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_goal_conflicts(self) -> List[ConflictInfo]:
        """检测目标冲突"""
        conflicts = []
        
        # 简化的目标冲突检测
        # 实际实现中需要更复杂的逻辑
        
        return conflicts
    
    def _detect_strategy_conflicts(self) -> List[ConflictInfo]:
        """检测策略冲突"""
        conflicts = []
        
        # 检查相邻层次间的策略冲突
        for i in range(len(self.levels) - 1):
            current_level = self.levels[i]
            next_level = self.levels[i + 1]
            
            current_nodes = [n for n in self.learning_nodes.values() if n.level == current_level]
            next_nodes = [n for n in self.learning_nodes.values() if n.level == next_level]
            
            # 简化的策略冲突检测
            if len(current_nodes) > 0 and len(next_nodes) > 0:
                # 检查是否有不兼容的策略配置
                for current_node in current_nodes:
                    for next_node in next_nodes:
                        if self._strategies_conflict(current_node, next_node):
                            conflict = ConflictInfo(
                                conflict_id=str(uuid.uuid4()),
                                conflict_type=ConflictType.STRATEGY_CONFLICT,
                                involved_nodes=[current_node.id, next_node.id],
                                description=f"策略冲突: {current_node.id} 与 {next_node.id}",
                                severity=5
                            )
                            conflicts.append(conflict)
        
        return conflicts
    
    def _strategies_conflict(self, node1: LearningNode, node2: LearningNode) -> bool:
        """检查两个节点的策略是否冲突"""
        # 简化的策略冲突检测
        # 实际实现中需要更复杂的策略兼容性检查
        
        # 如果两个节点都处于冲突状态，认为存在策略冲突
        return (node1.state == LearningState.CONFLICTING and 
                node2.state == LearningState.CONFLICTING)
    
    def _resolve_conflict(self, conflict: ConflictInfo) -> Dict[str, Any]:
        """解决冲突"""
        resolution = {
            "conflict_id": conflict.conflict_id,
            "resolved": False,
            "strategy": "",
            "actions": [],
            "success_rate": 0.0
        }
        
        if conflict.conflict_type == ConflictType.RESOURCE_CONFLICT:
            resolution.update(self._resolve_resource_conflict(conflict))
        elif conflict.conflict_type == ConflictType.STRATEGY_CONFLICT:
            resolution.update(self._resolve_strategy_conflict(conflict))
        else:
            # 默认解决策略
            resolution["strategy"] = "default_resolution"
            resolution["actions"] = ["重新协调", "重新分配资源"]
            resolution["success_rate"] = 0.7
        
        if resolution["resolved"]:
            conflict.resolved = True
            self.logger.info(f"解决冲突: {conflict.conflict_id}")
        
        return resolution
    
    def _resolve_resource_conflict(self, conflict: ConflictInfo) -> Dict[str, Any]:
        """解决资源冲突"""
        resolution = {
            "strategy": "resource_rebalancing",
            "actions": [],
            "success_rate": 0.8
        }
        
        # 重新分配资源
        level = None
        for node_id in conflict.involved_nodes:
            if node_id in self.learning_nodes:
                level = self.learning_nodes[node_id].level
                break
        
        if level:
            # 移动部分节点到其他层次
            level_manager = self.level_managers[level]
            moved_nodes = level_manager.redistribute_resources()
            
            resolution["actions"] = [f"重新分配 {len(moved_nodes)} 个节点"]
            resolution["resolved"] = len(moved_nodes) > 0
        
        return resolution
    
    def _resolve_strategy_conflict(self, conflict: ConflictInfo) -> Dict[str, Any]:
        """解决策略冲突"""
        resolution = {
            "strategy": "strategy_harmonization",
            "actions": [],
            "success_rate": 0.6
        }
        
        # 协调策略
        for node_id in conflict.involved_nodes:
            if node_id in self.learning_nodes:
                node = self.learning_nodes[node_id]
                node.state = LearningState.IDLE  # 重置状态
                resolution["actions"].append(f"重置节点 {node_id} 状态")
        
        resolution["resolved"] = True
        return resolution
    
    def monitor_learning_states(self) -> Dict[str, Any]:
        """
        监控学习层次状态
        
        Returns:
            监控结果
        """
        monitoring_results = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "level_status": {},
            "performance_trends": {},
            "alerts": []
        }
        
        with self.lock:
            # 1. 监控各层次状态
            level_status = {}
            for level in self.levels:
                level_nodes = [n for n in self.learning_nodes.values() if n.level == level]
                
                if not level_nodes:
                    status = "empty"
                    state_counts = {}
                else:
                    # 计算状态分布
                    state_counts = defaultdict(int)
                    for node in level_nodes:
                        state_counts[node.state] += 1
                    
                    if state_counts[LearningState.CONFLICTING] > 0:
                        status = "conflicted"
                    elif state_counts[LearningState.LEARNING] > 0:
                        status = "active"
                    else:
                        status = "idle"
                
                level_status[level.value] = {
                    "status": status,
                    "node_count": len(level_nodes),
                    "state_distribution": dict(state_counts)
                }
            
            monitoring_results["level_status"] = level_status
            
            # 2. 分析性能趋势
            performance_trends = self._analyze_performance_trends()
            monitoring_results["performance_trends"] = performance_trends
            
            # 3. 生成告警
            alerts = self._generate_alerts(level_status, performance_trends)
            monitoring_results["alerts"] = alerts
            
            # 4. 确定整体状态
            if alerts:
                monitoring_results["overall_status"] = "warning"
            if any(level_status[level]["status"] == "conflicted" for level in level_status):
                monitoring_results["overall_status"] = "critical"
        
        return monitoring_results
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """分析性能趋势"""
        trends = {}
        
        for level in self.levels:
            level_key = level.value
            if level_key in self.performance_history:
                history = list(self.performance_history[level_key])
                if len(history) >= 2:
                    recent_avg = np.mean(history[-5:]) if len(history) >= 5 else np.mean(history)
                    older_avg = np.mean(history[:-5]) if len(history) >= 10 else np.mean(history)
                    
                    trend_direction = "improving" if recent_avg > older_avg else "declining"
                    trend_magnitude = abs(recent_avg - older_avg)
                    
                    trends[level_key] = {
                        "direction": trend_direction,
                        "magnitude": trend_magnitude,
                        "current_value": recent_avg
                    }
        
        return trends
    
    def _generate_alerts(self, level_status: Dict[str, Any], 
                        performance_trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成告警"""
        alerts = []
        
        # 检查空层次
        for level, status_info in level_status.items():
            if status_info["status"] == "empty":
                alerts.append({
                    "type": "empty_level",
                    "level": level,
                    "severity": "low",
                    "message": f"层次 {level} 没有活跃节点"
                })
            
            if status_info["status"] == "conflicted":
                alerts.append({
                    "type": "conflicted_level",
                    "level": level,
                    "severity": "high",
                    "message": f"层次 {level} 存在冲突"
                })
        
        # 检查性能下降
        for level, trend_info in performance_trends.items():
            if trend_info["direction"] == "declining" and trend_info["magnitude"] > 0.2:
                alerts.append({
                    "type": "performance_decline",
                    "level": level,
                    "severity": "medium",
                    "message": f"层次 {level} 性能显著下降"
                })
        
        return alerts
    
    def optimize_learning_strategies(self) -> Dict[str, Any]:
        """
        优化学习策略
        
        Returns:
            策略优化结果
        """
        self.logger.info("开始学习策略优化")
        
        optimization_results = {
            "timestamp": time.time(),
            "strategy_evaluations": {},
            "optimized_strategies": {},
            "performance_predictions": {},
            "recommendations": []
        }
        
        with self.lock:
            # 1. 评估当前策略
            strategy_evaluations = self._evaluate_current_strategies()
            optimization_results["strategy_evaluations"] = strategy_evaluations
            
            # 2. 优化策略
            optimized_strategies = self._optimize_strategies(strategy_evaluations)
            optimization_results["optimized_strategies"] = optimized_strategies
            
            # 3. 预测性能
            performance_predictions = self._predict_strategy_performance(optimized_strategies)
            optimization_results["performance_predictions"] = performance_predictions
            
            # 4. 生成建议
            recommendations = self._generate_strategy_recommendations(optimized_strategies, performance_predictions)
            optimization_results["recommendations"] = recommendations
        
        self.logger.info(f"策略优化完成，优化了 {len(optimized_strategies)} 个策略")
        return optimization_results
    
    def _evaluate_current_strategies(self) -> Dict[str, float]:
        """评估当前策略"""
        evaluations = {}
        
        # 简化的策略评估
        # 实际实现中需要更复杂的策略效果评估
        
        for level in self.levels:
            level_key = level.value
            
            # 基于节点性能评估策略效果
            level_nodes = [n for n in self.learning_nodes.values() if n.level == level]
            
            if level_nodes:
                avg_performance = np.mean([
                    n.performance_metrics.get('overall_score', 0.5) 
                    for n in level_nodes
                ])
                evaluations[level_key] = avg_performance
            else:
                evaluations[level_key] = 0.3  # 默认分数
        
        return evaluations
    
    def _optimize_strategies(self, evaluations: Dict[str, float]) -> Dict[str, Any]:
        """优化策略"""
        optimized = {}
        
        for level, score in evaluations.items():
            if score < 0.6:
                # 需要优化
                optimization = {
                    "original_score": score,
                    "optimization_type": "performance_enhancement",
                    "improvement_actions": [],
                    "expected_improvement": 0.0
                }
                
                # 生成优化动作
                if score < 0.4:
                    optimization["improvement_actions"].append("重新设计学习算法")
                    optimization["improvement_actions"].append("增加训练数据")
                    optimization["expected_improvement"] = 0.3
                else:
                    optimization["improvement_actions"].append("微调参数")
                    optimization["improvement_actions"].append("优化特征提取")
                    optimization["expected_improvement"] = 0.15
                
                optimized[level] = optimization
        
        return optimized
    
    def _predict_strategy_performance(self, optimized_strategies: Dict[str, Any]) -> Dict[str, float]:
        """预测策略性能"""
        predictions = {}
        
        for level, optimization in optimized_strategies.items():
            original_score = optimization["original_score"]
            expected_improvement = optimization["expected_improvement"]
            
            # 简化的性能预测
            predicted_score = min(1.0, original_score + expected_improvement)
            predictions[level] = predicted_score
        
        return predictions
    
    def _generate_strategy_recommendations(self, optimized_strategies: Dict[str, Any], 
                                         predictions: Dict[str, float]) -> List[str]:
        """生成策略建议"""
        recommendations = []
        
        if optimized_strategies:
            recommendations.append(f"实施 {len(optimized_strategies)} 个策略优化")
            
            for level, optimization in optimized_strategies.items():
                actions = optimization["improvement_actions"]
                recommendations.append(f"层次 {level}: {'; '.join(actions)}")
        
        # 基于预测结果的建议
        high_potential_levels = [level for level, pred in predictions.items() if pred > 0.8]
        if high_potential_levels:
            recommendations.append(f"重点关注高性能潜力层次: {', '.join(high_potential_levels)}")
        
        return recommendations
    
    def start_monitoring(self):
        """启动监控"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("启动学习状态监控")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("停止学习状态监控")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集性能数据
                self._collect_performance_data()
                
                # 检查异常状态
                self._check_abnormal_states()
                
                time.sleep(10)  # 每10秒检查一次
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(30)  # 错误时延长等待时间
    
    def _collect_performance_data(self):
        """收集性能数据"""
        with self.lock:
            for level in self.levels:
                level_nodes = [n for n in self.learning_nodes.values() if n.level == level]
                
                if level_nodes:
                    avg_performance = np.mean([
                        n.performance_metrics.get('overall_score', 0.5) 
                        for n in level_nodes
                    ])
                    self.performance_history[level.value].append(avg_performance)
    
    def _check_abnormal_states(self):
        """检查异常状态"""
        abnormal_nodes = [
            node for node in self.learning_nodes.values() 
            if node.state == LearningState.CONFLICTING
        ]
        
        if abnormal_nodes:
            self.logger.warning(f"检测到 {len(abnormal_nodes)} 个异常节点")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            系统状态信息
        """
        with self.lock:
            status = {
                "timestamp": time.time(),
                "total_nodes": len(self.learning_nodes),
                "total_transfers": len(self.learning_transfers),
                "active_conflicts": len([c for c in self.conflicts.values() if not c.resolved]),
                "level_distribution": {},
                "monitoring_active": self.monitoring_active,
                "system_health": "healthy"
            }
            
            # 层次分布
            for level in self.levels:
                level_nodes = [n for n in self.learning_nodes.values() if n.level == level]
                status["level_distribution"][level.value] = len(level_nodes)
            
            # 系统健康度评估
            if status["active_conflicts"] > 0:
                status["system_health"] = "warning"
            if status["active_conflicts"] > 5:
                status["system_health"] = "critical"
            
            return status
    
    def shutdown(self):
        """关闭管理器"""
        self.logger.info("关闭层次学习管理器")
        
        # 停止监控
        self.stop_monitoring()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        self.logger.info("层次学习管理器已关闭")


class LevelManager:
    """层次管理器"""
    
    def __init__(self, level: LearningLevel, manager: HierarchicalLearningManager):
        self.level = level
        self.manager = manager
        self.logger = logging.getLogger(f'LevelManager_{level.value}')
    
    def process_transfer(self, transfer: LearningTransfer):
        """处理学习传递"""
        self.logger.info(f"处理学习传递到层次 {self.level.value}")
        
        # 模拟处理过程
        time.sleep(0.1)
        
        # 更新相关节点状态
        self._update_node_states(transfer)
    
    def _update_node_states(self, transfer: LearningTransfer):
        """更新节点状态"""
        # 简化的状态更新逻辑
        pass
    
    def optimize_efficiency(self) -> List[str]:
        """优化效率"""
        actions = []
        
        # 模拟效率优化
        actions.append("优化算法复杂度")
        actions.append("减少不必要的计算")
        
        return actions
    
    def optimize_quality(self) -> List[str]:
        """优化质量"""
        actions = []
        
        # 模拟质量优化
        actions.append("改进数据预处理")
        actions.append("增强特征提取")
        
        return actions
    
    def redistribute_resources(self) -> List[str]:
        """重新分配资源"""
        moved_nodes = []
        
        # 模拟资源重新分配
        # 实际实现中需要更复杂的逻辑
        
        return moved_nodes


# 使用示例
if __name__ == "__main__":
    # 创建层次学习管理器
    manager = HierarchicalLearningManager()
    
    try:
        # 创建学习节点
        feature_node = manager.create_learning_node(LearningLevel.FEATURE)
        pattern_node = manager.create_learning_node(LearningLevel.PATTERN)
        
        # 连接节点
        manager.connect_nodes(feature_node, pattern_node)
        
        # 执行学习传递
        transfer_id = manager.transfer_learning(
            LearningLevel.FEATURE, 
            LearningLevel.PATTERN, 
            {"features": [1, 2, 3, 4, 5]}
        )
        
        # 优化层次
        optimization_result = manager.optimize_learning_hierarchy()
        print(f"优化结果: {optimization_result}")
        
        # 评估跨层次效果
        evaluation_result = manager.evaluate_cross_level_effectiveness()
        print(f"评估结果: {evaluation_result}")
        
        # 监控状态
        monitoring_result = manager.monitor_learning_states()
        print(f"监控结果: {monitoring_result}")
        
        # 获取系统状态
        system_status = manager.get_system_status()
        print(f"系统状态: {system_status}")
        
    finally:
        # 关闭管理器
        manager.shutdown()