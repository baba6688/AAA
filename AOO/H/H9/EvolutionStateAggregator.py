# -*- coding: utf-8 -*-
"""
H9 进化状态聚合器
Evolution State Aggregator

功能模块：
1. 多模块进化状态融合
2. 进化状态评估和量化
3. 进化一致性检验和验证
4. 进化优先级排序和选择
5. 进化状态历史记录
6. 进化状态报告生成
7. 进化状态预警机制

版本: 1.0.0
作者: MiniMax Agent
日期: 2025-11-05
"""

import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
from enum import Enum
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvolutionStatus(Enum):
    """进化状态枚举"""
    INITIALIZING = "初始化中"
    ACTIVE = "活跃"
    STABLE = "稳定"
    EVOLVING = "进化中"
    CONVERGING = "收敛中"
    DIVERGING = "发散中"
    CRITICAL = "临界状态"
    FAILED = "失败"


class PriorityLevel(Enum):
    """优先级级别"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class ModuleState:
    """模块状态数据类"""
    module_id: str
    module_name: str
    status: EvolutionStatus
    health_score: float
    performance_score: float
    evolution_score: float
    timestamp: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """数据验证"""
        if not 0 <= self.health_score <= 1:
            warnings.warn(f"健康分数应在0-1之间: {self.health_score}")
        if not 0 <= self.performance_score <= 1:
            warnings.warn(f"性能分数应在0-1之间: {self.performance_score}")
        if not 0 <= self.evolution_score <= 1:
            warnings.warn(f"进化分数应在0-1之间: {self.evolution_score}")


@dataclass
class EvolutionMetrics:
    """进化指标数据类"""
    convergence_rate: float
    diversity_index: float
    stability_index: float
    innovation_index: float
    efficiency_score: float
    risk_score: float
    timestamp: float


@dataclass
class EvolutionReport:
    """进化报告数据类"""
    report_id: str
    timestamp: float
    overall_status: EvolutionStatus
    aggregated_score: float
    module_states: List[ModuleState]
    evolution_metrics: EvolutionMetrics
    recommendations: List[str]
    warnings: List[str]
    priority_actions: List[Dict[str, Any]]


class EvolutionStateAggregator:
    """进化状态聚合器"""
    
    def __init__(self, 
                 history_size: int = 1000,
                 aggregation_weights: Optional[Dict[str, float]] = None,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        初始化进化状态聚合器
        
        Args:
            history_size: 历史记录大小
            aggregation_weights: 聚合权重配置
            alert_thresholds: 预警阈值配置
        """
        self.history_size = history_size
        self.module_states: Dict[str, ModuleState] = {}
        self.evolution_history: deque = deque(maxlen=history_size)
        self.aggregation_weights = aggregation_weights or {
            'health_score': 0.3,
            'performance_score': 0.4,
            'evolution_score': 0.3
        }
        self.alert_thresholds = alert_thresholds or {
            'critical_health': 0.3,
            'low_performance': 0.4,
            'high_risk': 0.7,
            'diversity_threshold': 0.2
        }
        
        self.lock = threading.RLock()
        self.is_running = False
        self.aggregation_thread = None
        
        # 统计信息
        self.stats = {
            'total_aggregations': 0,
            'successful_aggregations': 0,
            'failed_aggregations': 0,
            'alerts_generated': 0,
            'reports_generated': 0
        }
        
        logger.info("进化状态聚合器初始化完成")
    
    def register_module(self, module_state: ModuleState) -> bool:
        """
        注册模块状态
        
        Args:
            module_state: 模块状态对象
            
        Returns:
            bool: 注册是否成功
        """
        try:
            with self.lock:
                self.module_states[module_state.module_id] = module_state
                logger.info(f"模块 {module_state.module_name} 注册成功")
                return True
        except Exception as e:
            logger.error(f"模块注册失败: {e}")
            return False
    
    def unregister_module(self, module_id: str) -> bool:
        """
        注销模块
        
        Args:
            module_id: 模块ID
            
        Returns:
            bool: 注销是否成功
        """
        try:
            with self.lock:
                if module_id in self.module_states:
                    module_name = self.module_states[module_id].module_name
                    del self.module_states[module_id]
                    logger.info(f"模块 {module_name} 注销成功")
                    return True
                else:
                    logger.warning(f"模块 {module_id} 不存在")
                    return False
        except Exception as e:
            logger.error(f"模块注销失败: {e}")
            return False
    
    def update_module_state(self, module_id: str, **kwargs) -> bool:
        """
        更新模块状态
        
        Args:
            module_id: 模块ID
            **kwargs: 更新参数
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with self.lock:
                if module_id not in self.module_states:
                    logger.error(f"模块 {module_id} 未注册")
                    return False
                
                module_state = self.module_states[module_id]
                for key, value in kwargs.items():
                    if hasattr(module_state, key):
                        setattr(module_state, key, value)
                
                module_state.timestamp = time.time()
                logger.debug(f"模块 {module_id} 状态更新成功")
                return True
        except Exception as e:
            logger.error(f"模块状态更新失败: {e}")
            return False
    
    def aggregate_evolution_states(self) -> EvolutionMetrics:
        """
        聚合进化状态
        
        Returns:
            EvolutionMetrics: 进化指标
        """
        try:
            with self.lock:
                if not self.module_states:
                    logger.warning("没有注册的模块")
                    return self._create_default_metrics()
                
                # 计算各项指标
                convergence_rate = self._calculate_convergence_rate()
                diversity_index = self._calculate_diversity_index()
                stability_index = self._calculate_stability_index()
                innovation_index = self._calculate_innovation_index()
                efficiency_score = self._calculate_efficiency_score()
                risk_score = self._calculate_risk_score()
                
                metrics = EvolutionMetrics(
                    convergence_rate=convergence_rate,
                    diversity_index=diversity_index,
                    stability_index=stability_index,
                    innovation_index=innovation_index,
                    efficiency_score=efficiency_score,
                    risk_score=risk_score,
                    timestamp=time.time()
                )
                
                self.evolution_history.append(metrics)
                self.stats['total_aggregations'] += 1
                self.stats['successful_aggregations'] += 1
                
                return metrics
                
        except Exception as e:
            logger.error(f"进化状态聚合失败: {e}")
            self.stats['failed_aggregations'] += 1
            return self._create_default_metrics()
    
    def _calculate_convergence_rate(self) -> float:
        """计算收敛率"""
        if len(self.evolution_history) < 2:
            return 0.5
        
        recent_metrics = list(self.evolution_history)[-10:]
        scores = [m.efficiency_score for m in recent_metrics]
        
        if len(scores) < 2:
            return 0.5
        
        # 计算趋势稳定性
        diffs = np.diff(scores)
        stability = 1.0 - np.std(diffs) if len(diffs) > 0 else 0.5
        
        return max(0.0, min(1.0, stability))
    
    def _calculate_diversity_index(self) -> float:
        """计算多样性指数"""
        if not self.module_states:
            return 0.0
        
        # 计算模块间差异
        health_scores = [state.health_score for state in self.module_states.values()]
        performance_scores = [state.performance_score for state in self.module_states.values()]
        
        health_diversity = np.std(health_scores) if len(health_scores) > 1 else 0.0
        performance_diversity = np.std(performance_scores) if len(performance_scores) > 1 else 0.0
        
        # 归一化多样性
        diversity = (health_diversity + performance_diversity) / 2.0
        return max(0.0, min(1.0, diversity))
    
    def _calculate_stability_index(self) -> float:
        """计算稳定性指数"""
        if len(self.evolution_history) < 5:
            return 0.5
        
        recent_metrics = list(self.evolution_history)[-5:]
        scores = [m.stability_index for m in recent_metrics]
        
        # 计算稳定性趋势
        stability = np.mean(scores) if scores else 0.5
        return max(0.0, min(1.0, stability))
    
    def _calculate_innovation_index(self) -> float:
        """计算创新指数"""
        if not self.module_states:
            return 0.0
        
        # 基于进化分数和性能提升计算创新性
        evolution_scores = [state.evolution_score for state in self.module_states.values()]
        performance_scores = [state.performance_score for state in self.module_states.values()]
        
        avg_evolution = np.mean(evolution_scores)
        avg_performance = np.mean(performance_scores)
        
        # 创新指数 = 进化活跃度 * 性能提升潜力
        innovation = avg_evolution * (1.0 - avg_performance)
        return max(0.0, min(1.0, innovation))
    
    def _calculate_efficiency_score(self) -> float:
        """计算效率分数"""
        if not self.module_states:
            return 0.0
        
        # 加权计算整体效率
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for state in self.module_states.values():
            weight = self.aggregation_weights.get('health_score', 0.3) * state.health_score + \
                    self.aggregation_weights.get('performance_score', 0.4) * state.performance_score + \
                    self.aggregation_weights.get('evolution_score', 0.3) * state.evolution_score
            
            total_weighted_score += weight
            total_weight += 1.0
        
        efficiency = total_weighted_score / total_weight if total_weight > 0 else 0.0
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_risk_score(self) -> float:
        """计算风险分数"""
        if not self.module_states:
            return 0.0
        
        risk_factors = []
        
        # 健康风险
        health_scores = [state.health_score for state in self.module_states.values()]
        min_health = min(health_scores) if health_scores else 1.0
        health_risk = 1.0 - min_health
        
        # 性能风险
        performance_scores = [state.performance_score for state in self.module_states.values()]
        min_performance = min(performance_scores) if performance_scores else 1.0
        performance_risk = 1.0 - min_performance
        
        # 状态风险
        status_risks = {
            EvolutionStatus.CRITICAL: 0.9,
            EvolutionStatus.FAILED: 1.0,
            EvolutionStatus.DIVERGING: 0.7,
            EvolutionStatus.CONVERGING: 0.3,
            EvolutionStatus.EVOLVING: 0.4,
            EvolutionStatus.ACTIVE: 0.1,
            EvolutionStatus.STABLE: 0.05,
            EvolutionStatus.INITIALIZING: 0.2
        }
        
        status_risk = np.mean([status_risks.get(state.status, 0.5) 
                              for state in self.module_states.values()])
        
        # 综合风险
        total_risk = (health_risk * 0.4 + performance_risk * 0.3 + status_risk * 0.3)
        return max(0.0, min(1.0, total_risk))
    
    def _create_default_metrics(self) -> EvolutionMetrics:
        """创建默认指标"""
        return EvolutionMetrics(
            convergence_rate=0.5,
            diversity_index=0.5,
            stability_index=0.5,
            innovation_index=0.5,
            efficiency_score=0.5,
            risk_score=0.5,
            timestamp=time.time()
        )
    
    def validate_evolution_consistency(self) -> Tuple[bool, List[str]]:
        """
        验证进化一致性
        
        Returns:
            Tuple[bool, List[str]]: (是否一致, 问题列表)
        """
        issues = []
        
        try:
            with self.lock:
                if not self.module_states:
                    return True, []
                
                # 检查健康分数一致性
                health_scores = [state.health_score for state in self.module_states.values()]
                if np.std(health_scores) > 0.5:
                    issues.append("模块健康分数差异过大")
                
                # 检查性能分数一致性
                performance_scores = [state.performance_score for state in self.module_states.values()]
                if np.std(performance_scores) > 0.5:
                    issues.append("模块性能分数差异过大")
                
                # 检查状态一致性
                statuses = [state.status for state in self.module_states.values()]
                critical_modules = [s for s in statuses if s in [EvolutionStatus.CRITICAL, EvolutionStatus.FAILED]]
                if critical_modules and len(critical_modules) > len(statuses) * 0.3:
                    issues.append("关键状态模块比例过高")
                
                # 检查时间戳一致性
                current_time = time.time()
                stale_modules = [state for state in self.module_states.values() 
                               if current_time - state.timestamp > 300]  # 5分钟
                if stale_modules:
                    issues.append(f"存在 {len(stale_modules)} 个过时模块状态")
                
                is_consistent = len(issues) == 0
                return is_consistent, issues
                
        except Exception as e:
            logger.error(f"进化一致性验证失败: {e}")
            return False, [f"验证过程出错: {str(e)}"]
    
    def prioritize_evolution_actions(self) -> List[Dict[str, Any]]:
        """
        优先级排序和选择进化动作
        
        Returns:
            List[Dict[str, Any]]: 优先级排序后的动作列表
        """
        actions = []
        
        try:
            with self.lock:
                # 基于当前状态生成动作
                for module_id, state in self.module_states.items():
                    action = self._generate_module_action(module_id, state)
                    if action:
                        actions.append(action)
                
                # 按优先级排序
                actions.sort(key=lambda x: x['priority'], reverse=True)
                
                # 限制动作数量
                max_actions = 10
                return actions[:max_actions]
                
        except Exception as e:
            logger.error(f"进化动作优先级排序失败: {e}")
            return []
    
    def _generate_module_action(self, module_id: str, state: ModuleState) -> Optional[Dict[str, Any]]:
        """为模块生成动作"""
        action = {
            'module_id': module_id,
            'module_name': state.module_name,
            'current_status': state.status.value,
            'priority': PriorityLevel.MEDIUM.value,
            'action_type': 'monitor',
            'description': '持续监控',
            'timestamp': time.time()
        }
        
        # 基于状态生成相应动作
        if state.health_score < self.alert_thresholds['critical_health']:
            action.update({
                'priority': PriorityLevel.CRITICAL.value,
                'action_type': 'health_check',
                'description': f"健康分数过低 ({state.health_score:.2f})，需要立即检查"
            })
        elif state.performance_score < self.alert_thresholds['low_performance']:
            action.update({
                'priority': PriorityLevel.HIGH.value,
                'action_type': 'performance_optimize',
                'description': f"性能分数偏低 ({state.performance_score:.2f})，需要优化"
            })
        elif state.status == EvolutionStatus.FAILED:
            action.update({
                'priority': PriorityLevel.EMERGENCY.value,
                'action_type': 'recovery',
                'description': "模块失败，需要立即恢复"
            })
        elif state.evolution_score > 0.8:
            action.update({
                'priority': PriorityLevel.HIGH.value,
                'action_type': 'evolution_accelerate',
                'description': f"进化分数很高 ({state.evolution_score:.2f})，可以加速进化"
            })
        
        return action
    
    def generate_evolution_report(self) -> EvolutionReport:
        """
        生成进化状态报告
        
        Returns:
            EvolutionReport: 进化报告
        """
        try:
            with self.lock:
                # 聚合当前状态
                metrics = self.aggregate_evolution_states()
                
                # 计算整体状态
                overall_status = self._determine_overall_status(metrics)
                
                # 计算聚合分数
                aggregated_score = self._calculate_aggregated_score()
                
                # 生成建议
                recommendations = self._generate_recommendations(metrics)
                
                # 生成预警
                warnings = self._generate_warnings(metrics)
                
                # 生成优先级动作
                priority_actions = self.prioritize_evolution_actions()
                
                report = EvolutionReport(
                    report_id=f"evo_report_{int(time.time())}",
                    timestamp=time.time(),
                    overall_status=overall_status,
                    aggregated_score=aggregated_score,
                    module_states=list(self.module_states.values()),
                    evolution_metrics=metrics,
                    recommendations=recommendations,
                    warnings=warnings,
                    priority_actions=priority_actions
                )
                
                self.stats['reports_generated'] += 1
                logger.info(f"进化报告生成成功: {report.report_id}")
                return report
                
        except Exception as e:
            logger.error(f"进化报告生成失败: {e}")
            # 返回默认报告
            return EvolutionReport(
                report_id=f"error_report_{int(time.time())}",
                timestamp=time.time(),
                overall_status=EvolutionStatus.CRITICAL,
                aggregated_score=0.0,
                module_states=[],
                evolution_metrics=self._create_default_metrics(),
                recommendations=["系统错误，无法生成正常报告"],
                warnings=["进化状态聚合器出现异常"],
                priority_actions=[]
            )
    
    def _determine_overall_status(self, metrics: EvolutionMetrics) -> EvolutionStatus:
        """确定整体状态"""
        if metrics.risk_score > 0.8:
            return EvolutionStatus.CRITICAL
        elif metrics.efficiency_score < 0.3:
            return EvolutionStatus.FAILED
        elif metrics.stability_index < 0.4:
            return EvolutionStatus.DIVERGING
        elif metrics.convergence_rate > 0.8:
            return EvolutionStatus.CONVERGING
        elif metrics.innovation_index > 0.7:
            return EvolutionStatus.EVOLVING
        elif metrics.efficiency_score > 0.7:
            return EvolutionStatus.ACTIVE
        else:
            return EvolutionStatus.STABLE
    
    def _calculate_aggregated_score(self) -> float:
        """计算聚合分数"""
        if not self.module_states:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for state in self.module_states.values():
            weight = (self.aggregation_weights['health_score'] * state.health_score +
                     self.aggregation_weights['performance_score'] * state.performance_score +
                     self.aggregation_weights['evolution_score'] * state.evolution_score)
            
            total_score += weight
            total_weight += 1.0
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, metrics: EvolutionMetrics) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if metrics.risk_score > 0.7:
            recommendations.append("风险分数过高，建议立即进行系统检查")
        
        if metrics.diversity_index < self.alert_thresholds['diversity_threshold']:
            recommendations.append("多样性不足，建议增加模块间的差异化")
        
        if metrics.convergence_rate < 0.4:
            recommendations.append("收敛率较低，建议优化算法参数")
        
        if metrics.innovation_index > 0.8:
            recommendations.append("创新指数很高，建议加速进化进程")
        
        if metrics.efficiency_score < 0.5:
            recommendations.append("效率分数偏低，建议进行性能优化")
        
        if not recommendations:
            recommendations.append("系统运行正常，继续保持当前状态")
        
        return recommendations
    
    def _generate_warnings(self, metrics: EvolutionMetrics) -> List[str]:
        """生成预警"""
        warnings = []
        
        if metrics.risk_score > self.alert_thresholds['high_risk']:
            warnings.append(f"高风险预警：风险分数达到 {metrics.risk_score:.2f}")
        
        if metrics.stability_index < 0.3:
            warnings.append("稳定性预警：系统稳定性较差")
        
        if metrics.convergence_rate < 0.2:
            warnings.append("收敛性预警：系统难以收敛")
        
        # 检查模块状态
        critical_modules = [state for state in self.module_states.values() 
                          if state.status in [EvolutionStatus.CRITICAL, EvolutionStatus.FAILED]]
        if critical_modules:
            warnings.append(f"模块预警：{len(critical_modules)} 个模块处于关键状态")
        
        return warnings
    
    def check_evolution_alerts(self) -> List[Dict[str, Any]]:
        """
        检查进化状态预警
        
        Returns:
            List[Dict[str, Any]]: 预警列表
        """
        alerts = []
        
        try:
            with self.lock:
                metrics = self.aggregate_evolution_states()
                
                # 风险预警
                if metrics.risk_score > self.alert_thresholds['high_risk']:
                    alerts.append({
                        'type': 'risk',
                        'level': 'high',
                        'message': f'风险分数过高: {metrics.risk_score:.2f}',
                        'timestamp': time.time(),
                        'action_required': True
                    })
                
                # 健康预警
                if self.module_states:
                    min_health = min(state.health_score for state in self.module_states.values())
                    if min_health < self.alert_thresholds['critical_health']:
                        alerts.append({
                            'type': 'health',
                            'level': 'critical',
                            'message': f'最低健康分数过低: {min_health:.2f}',
                            'timestamp': time.time(),
                            'action_required': True
                        })
                
                # 多样性预警
                if metrics.diversity_index < self.alert_thresholds['diversity_threshold']:
                    alerts.append({
                        'type': 'diversity',
                        'level': 'medium',
                        'message': f'多样性不足: {metrics.diversity_index:.2f}',
                        'timestamp': time.time(),
                        'action_required': False
                    })
                
                # 状态预警
                failed_modules = [state for state in self.module_states.values() 
                                if state.status == EvolutionStatus.FAILED]
                if failed_modules:
                    alerts.append({
                        'type': 'status',
                        'level': 'emergency',
                        'message': f'{len(failed_modules)} 个模块处于失败状态',
                        'timestamp': time.time(),
                        'action_required': True
                    })
                
                self.stats['alerts_generated'] += len(alerts)
                return alerts
                
        except Exception as e:
            logger.error(f"预警检查失败: {e}")
            return [{
                'type': 'system',
                'level': 'error',
                'message': f'预警系统错误: {str(e)}',
                'timestamp': time.time(),
                'action_required': True
            }]
    
    def start_monitoring(self, interval: float = 30.0):
        """
        开始监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self.is_running:
            logger.warning("监控已经在运行中")
            return
        
        self.is_running = True
        self.aggregation_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.aggregation_thread.start()
        logger.info(f"进化状态监控已启动，间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=5)
        logger.info("进化状态监控已停止")
    
    def _monitoring_loop(self, interval: float):
        """监控循环"""
        while self.is_running:
            try:
                # 执行监控任务
                alerts = self.check_evolution_alerts()
                
                # 处理预警
                for alert in alerts:
                    self._handle_alert(alert)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(interval)
    
    def _handle_alert(self, alert: Dict[str, Any]):
        """处理预警"""
        logger.warning(f"预警: {alert['message']}")
        
        # 这里可以添加实际的预警处理逻辑
        # 比如发送通知、触发自动化响应等
        if alert.get('action_required', False):
            logger.error(f"需要立即处理预警: {alert['message']}")
    
    def get_evolution_history(self, limit: int = 100) -> List[EvolutionMetrics]:
        """
        获取进化历史记录
        
        Args:
            limit: 限制返回数量
            
        Returns:
            List[EvolutionMetrics]: 历史记录列表
        """
        with self.lock:
            history = list(self.evolution_history)
            return history[-limit:] if limit > 0 else history
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.lock:
            stats = self.stats.copy()
            stats.update({
                'registered_modules': len(self.module_states),
                'history_size': len(self.evolution_history),
                'is_monitoring': self.is_running
            })
            return stats
    
    def export_report(self, report: EvolutionReport, file_path: str) -> bool:
        """
        导出进化报告
        
        Args:
            report: 进化报告
            file_path: 文件路径
            
        Returns:
            bool: 导出是否成功
        """
        try:
            # 转换为可序列化的格式
            module_states_serializable = []
            for state in report.module_states:
                state_dict = asdict(state)
                state_dict['status'] = state.status.value  # 转换枚举为字符串
                module_states_serializable.append(state_dict)
            
            metrics_dict = asdict(report.evolution_metrics)
            
            report_dict = {
                'report_id': report.report_id,
                'timestamp': report.timestamp,
                'timestamp_str': datetime.fromtimestamp(report.timestamp).isoformat(),
                'overall_status': report.overall_status.value,
                'aggregated_score': report.aggregated_score,
                'module_states': module_states_serializable,
                'evolution_metrics': metrics_dict,
                'recommendations': report.recommendations,
                'warnings': report.warnings,
                'priority_actions': report.priority_actions
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"进化报告已导出到: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"报告导出失败: {e}")
            return False
    
    def reset(self):
        """重置聚合器"""
        with self.lock:
            self.module_states.clear()
            self.evolution_history.clear()
            self.stats = {
                'total_aggregations': 0,
                'successful_aggregations': 0,
                'failed_aggregations': 0,
                'alerts_generated': 0,
                'reports_generated': 0
            }
            logger.info("进化状态聚合器已重置")


# 使用示例和测试函数
def demo_evolution_state_aggregator():
    """演示进化状态聚合器功能"""
    print("=== H9 进化状态聚合器演示 ===")
    
    # 创建聚合器
    aggregator = EvolutionStateAggregator(history_size=100)
    
    # 注册模拟模块
    modules_data = [
        {
            'module_id': 'module_1',
            'module_name': '数据采集模块',
            'status': EvolutionStatus.ACTIVE,
            'health_score': 0.9,
            'performance_score': 0.85,
            'evolution_score': 0.7
        },
        {
            'module_id': 'module_2',
            'module_name': '分析引擎',
            'status': EvolutionStatus.EVOLVING,
            'health_score': 0.8,
            'performance_score': 0.9,
            'evolution_score': 0.8
        },
        {
            'module_id': 'module_3',
            'module_name': '决策模块',
            'status': EvolutionStatus.STABLE,
            'health_score': 0.95,
            'performance_score': 0.88,
            'evolution_score': 0.6
        }
    ]
    
    # 注册模块
    for module_data in modules_data:
        module_state = ModuleState(
            module_id=module_data['module_id'],
            module_name=module_data['module_name'],
            status=module_data['status'],
            health_score=module_data['health_score'],
            performance_score=module_data['performance_score'],
            evolution_score=module_data['evolution_score'],
            timestamp=time.time(),
            metadata={}
        )
        aggregator.register_module(module_state)
    
    print(f"已注册 {len(aggregator.module_states)} 个模块")
    
    # 聚合进化状态
    metrics = aggregator.aggregate_evolution_states()
    print(f"\n进化指标:")
    print(f"  收敛率: {metrics.convergence_rate:.3f}")
    print(f"  多样性指数: {metrics.diversity_index:.3f}")
    print(f"  稳定性指数: {metrics.stability_index:.3f}")
    print(f"  创新指数: {metrics.innovation_index:.3f}")
    print(f"  效率分数: {metrics.efficiency_score:.3f}")
    print(f"  风险分数: {metrics.risk_score:.3f}")
    
    # 验证一致性
    is_consistent, issues = aggregator.validate_evolution_consistency()
    print(f"\n进化一致性: {'一致' if is_consistent else '不一致'}")
    if issues:
        print("问题列表:")
        for issue in issues:
            print(f"  - {issue}")
    
    # 优先级排序
    actions = aggregator.prioritize_evolution_actions()
    print(f"\n优先级动作 ({len(actions)} 项):")
    for action in actions:
        print(f"  [{action['priority']}] {action['action_type']}: {action['description']}")
    
    # 检查预警
    alerts = aggregator.check_evolution_alerts()
    print(f"\n预警信息 ({len(alerts)} 项):")
    for alert in alerts:
        print(f"  [{alert['level']}] {alert['type']}: {alert['message']}")
    
    # 生成报告
    report = aggregator.generate_evolution_report()
    print(f"\n进化报告:")
    print(f"  报告ID: {report.report_id}")
    print(f"  整体状态: {report.overall_status.value}")
    print(f"  聚合分数: {report.aggregated_score:.3f}")
    print(f"  建议数量: {len(report.recommendations)}")
    print(f"  预警数量: {len(report.warnings)}")
    
    # 导出报告
    report_file = "/tmp/evolution_report.json"
    if aggregator.export_report(report, report_file):
        print(f"\n报告已导出到: {report_file}")
    
    # 统计信息
    stats = aggregator.get_statistics()
    print(f"\n统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    demo_evolution_state_aggregator()