"""
G9执行状态聚合器
实现多模块执行状态融合、评估、检验、排序、历史记录、报告和预警功能
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import queue
import statistics
import math


class ExecutionStatus(Enum):
    """执行状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    PAUSED = "paused"
    BLOCKED = "blocked"


class PriorityLevel(Enum):
    """优先级等级"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    BACKGROUND = 1


class ConsistencyStatus(Enum):
    """一致性状态"""
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"


@dataclass
class ExecutionMetrics:
    """执行指标"""
    execution_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    throughput: float = 0.0
    latency: float = 0.0
    resource_utilization: float = 0.0


@dataclass
class ExecutionState:
    """执行状态"""
    module_id: str
    task_id: str
    status: ExecutionStatus
    priority: PriorityLevel
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    dependencies: List[str] = field(default_factory=list)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    consistency_score: float = 0.0
    risk_level: float = 0.0


@dataclass
class StateHistory:
    """状态历史记录"""
    timestamp: datetime
    state: ExecutionState
    aggregated_metrics: ExecutionMetrics
    consistency_status: ConsistencyStatus
    priority_score: float


@dataclass
class StateReport:
    """状态报告"""
    report_id: str
    generation_time: datetime
    summary: Dict[str, Any]
    detailed_metrics: Dict[str, ExecutionMetrics]
    consistency_analysis: Dict[str, Any]
    priority_ranking: List[Tuple[str, float]]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]


class IntelligentFusionAlgorithm:
    """智能执行状态融合算法"""
    
    def __init__(self):
        self.weights = {
            'execution_time': 0.15,
            'success_rate': 0.25,
            'resource_usage': 0.20,
            'priority': 0.15,
            'consistency': 0.15,
            'risk_level': 0.10
        }
    
    def fuse_states(self, states: List[ExecutionState]) -> ExecutionState:
        """融合多个执行状态"""
        if not states:
            raise ValueError("状态列表不能为空")
        
        if len(states) == 1:
            return states[0]
        
        # 计算融合指标
        fused_metrics = self._calculate_fused_metrics(states)
        fused_priority = self._calculate_fused_priority(states)
        fused_consistency = self._calculate_consistency_score(states)
        fused_risk = self._calculate_risk_score(states)
        
        # 创建融合状态
        fused_state = ExecutionState(
            module_id="aggregated",
            task_id="fused_task",
            status=self._determine_fused_status(states),
            priority=fused_priority,
            metrics=fused_metrics,
            consistency_score=fused_consistency,
            risk_level=fused_risk
        )
        
        return fused_state
    
    def _calculate_fused_metrics(self, states: List[ExecutionState]) -> ExecutionMetrics:
        """计算融合指标"""
        total_time = sum(s.metrics.execution_time for s in states)
        avg_cpu = statistics.mean(s.metrics.cpu_usage for s in states)
        avg_memory = statistics.mean(s.metrics.memory_usage for s in states)
        avg_success_rate = statistics.mean(s.metrics.success_rate for s in states)
        total_errors = sum(s.metrics.error_count for s in states)
        avg_throughput = statistics.mean(s.metrics.throughput for s in states)
        avg_latency = statistics.mean(s.metrics.latency for s in states)
        avg_resource = statistics.mean(s.metrics.resource_utilization for s in states)
        
        return ExecutionMetrics(
            execution_time=total_time,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            success_rate=avg_success_rate,
            error_count=total_errors,
            throughput=avg_throughput,
            latency=avg_latency,
            resource_utilization=avg_resource
        )
    
    def _calculate_fused_priority(self, states: List[ExecutionState]) -> PriorityLevel:
        """计算融合优先级"""
        priority_scores = {
            PriorityLevel.CRITICAL: 5,
            PriorityLevel.HIGH: 4,
            PriorityLevel.MEDIUM: 3,
            PriorityLevel.LOW: 2,
            PriorityLevel.BACKGROUND: 1
        }
        
        weighted_score = sum(
            priority_scores[s.priority] * len([t for t in states if t.priority == s.priority])
            for s in states
        ) / len(states)
        
        if weighted_score >= 4.5:
            return PriorityLevel.CRITICAL
        elif weighted_score >= 3.5:
            return PriorityLevel.HIGH
        elif weighted_score >= 2.5:
            return PriorityLevel.MEDIUM
        elif weighted_score >= 1.5:
            return PriorityLevel.LOW
        else:
            return PriorityLevel.BACKGROUND
    
    def _calculate_consistency_score(self, states: List[ExecutionState]) -> float:
        """计算一致性分数"""
        if len(states) <= 1:
            return 1.0
        
        # 检查状态一致性
        status_counts = defaultdict(int)
        for state in states:
            status_counts[state.status] += 1
        
        # 计算一致性分数
        max_count = max(status_counts.values())
        consistency_ratio = max_count / len(states)
        
        # 检查时间一致性
        time_consistency = self._check_time_consistency(states)
        
        # 检查资源使用一致性
        resource_consistency = self._check_resource_consistency(states)
        
        return (consistency_ratio + time_consistency + resource_consistency) / 3
    
    def _check_time_consistency(self, states: List[ExecutionState]) -> float:
        """检查时间一致性"""
        if len(states) <= 1:
            return 1.0
        
        execution_times = [s.metrics.execution_time for s in states if s.metrics.execution_time > 0]
        if not execution_times:
            return 0.5
        
        mean_time = statistics.mean(execution_times)
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        # 基于变异系数计算时间一致性
        cv = std_dev / mean_time if mean_time > 0 else 0
        return max(0, 1 - cv)
    
    def _check_resource_consistency(self, states: List[ExecutionState]) -> float:
        """检查资源使用一致性"""
        cpu_usage = [s.metrics.cpu_usage for s in states]
        memory_usage = [s.metrics.memory_usage for s in states]
        
        cpu_consistency = self._calculate_usage_consistency(cpu_usage)
        memory_consistency = self._calculate_usage_consistency(memory_usage)
        
        return (cpu_consistency + memory_consistency) / 2
    
    def _calculate_usage_consistency(self, usage_list: List[float]) -> float:
        """计算使用量一致性"""
        if not usage_list:
            return 0.5
        
        mean_usage = statistics.mean(usage_list)
        if mean_usage == 0:
            return 1.0
        
        std_dev = statistics.stdev(usage_list) if len(usage_list) > 1 else 0
        cv = std_dev / mean_usage
        
        return max(0, 1 - cv)
    
    def _calculate_risk_score(self, states: List[ExecutionState]) -> float:
        """计算风险分数"""
        risk_factors = []
        
        # 失败率风险
        failure_rate = sum(1 for s in states if s.status == ExecutionStatus.FAILED) / len(states)
        risk_factors.append(failure_rate)
        
        # 超时风险
        timeout_rate = sum(1 for s in states if s.status == ExecutionStatus.TIMEOUT) / len(states)
        risk_factors.append(timeout_rate)
        
        # 重试风险
        avg_retries = statistics.mean(s.retry_count for s in states)
        retry_risk = min(avg_retries / 3, 1.0)  # 标准化到0-1
        risk_factors.append(retry_risk)
        
        # 资源使用风险
        high_resource_usage = sum(
            1 for s in states 
            if s.metrics.cpu_usage > 80 or s.metrics.memory_usage > 80
        ) / len(states)
        risk_factors.append(high_resource_usage)
        
        return statistics.mean(risk_factors)
    
    def _determine_fused_status(self, states: List[ExecutionState]) -> ExecutionStatus:
        """确定融合状态"""
        status_counts = defaultdict(int)
        for state in states:
            status_counts[state.status] += 1
        
        # 优先级：失败 > 运行中 > 成功 > 其他
        if ExecutionStatus.FAILED in status_counts:
            return ExecutionStatus.FAILED
        elif ExecutionStatus.RUNNING in status_counts:
            return ExecutionStatus.RUNNING
        elif ExecutionStatus.SUCCESS in status_counts:
            return ExecutionStatus.SUCCESS
        else:
            # 返回最常见的状态
            return max(status_counts.items(), key=lambda x: x[1])[0]


class ExecutionStateAggregator:
    """执行状态聚合器主类"""
    
    def __init__(self, max_history_size: int = 1000, alert_threshold: float = 0.7):
        self.fusion_algorithm = IntelligentFusionAlgorithm()
        self.max_history_size = max_history_size
        self.alert_threshold = alert_threshold
        
        # 状态存储
        self.active_states: Dict[str, ExecutionState] = {}
        self.state_history: deque = deque(maxlen=max_history_size)
        self.module_states: Dict[str, List[ExecutionState]] = defaultdict(list)
        
        # 报告和预警
        self.reports: Dict[str, StateReport] = {}
        self.alerts: List[Dict[str, Any]] = []
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 日志配置
        self.logger = logging.getLogger(__name__)
        
        # 监控线程
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def register_module_state(self, module_id: str, state: ExecutionState) -> bool:
        """注册模块执行状态"""
        with self.lock:
            try:
                # 验证状态数据
                if not self._validate_state(state):
                    self.logger.warning(f"无效的执行状态: {module_id}")
                    return False
                
                # 更新活跃状态
                self.active_states[f"{module_id}_{state.task_id}"] = state
                
                # 更新模块状态历史
                self.module_states[module_id].append(state)
                if len(self.module_states[module_id]) > 100:  # 限制模块历史大小
                    self.module_states[module_id] = self.module_states[module_id][-50:]
                
                # 检查是否需要生成预警
                self._check_alerts(state)
                
                self.logger.info(f"成功注册模块状态: {module_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"注册模块状态失败: {e}")
                return False
    
    def aggregate_module_states(self, module_ids: List[str]) -> Optional[ExecutionState]:
        """聚合指定模块的状态"""
        with self.lock:
            try:
                states = []
                for module_id in module_ids:
                    if module_id in self.module_states and self.module_states[module_id]:
                        # 获取最新的状态
                        latest_state = self.module_states[module_id][-1]
                        states.append(latest_state)
                
                if not states:
                    self.logger.warning(f"没有找到模块状态: {module_ids}")
                    return None
                
                # 使用融合算法聚合状态
                fused_state = self.fusion_algorithm.fuse_states(states)
                
                # 记录历史
                self._record_history(fused_state)
                
                self.logger.info(f"成功聚合模块状态: {module_ids}")
                return fused_state
                
            except Exception as e:
                self.logger.error(f"聚合模块状态失败: {e}")
                return None
    
    def evaluate_execution_state(self, state: ExecutionState) -> Dict[str, float]:
        """评估执行状态"""
        try:
            evaluation = {}
            
            # 时间效率评估
            evaluation['time_efficiency'] = self._evaluate_time_efficiency(state)
            
            # 资源效率评估
            evaluation['resource_efficiency'] = self._evaluate_resource_efficiency(state)
            
            # 成功率评估
            evaluation['success_rate'] = state.metrics.success_rate
            
            # 优先级匹配度评估
            evaluation['priority_alignment'] = self._evaluate_priority_alignment(state)
            
            # 一致性评估
            evaluation['consistency'] = state.consistency_score
            
            # 风险评估
            evaluation['risk_score'] = state.risk_level
            
            # 综合评估分数
            evaluation['overall_score'] = self._calculate_overall_score(evaluation)
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"执行状态评估失败: {e}")
            return {}
    
    def check_consistency(self, states: List[ExecutionState]) -> ConsistencyStatus:
        """检查执行一致性"""
        try:
            if len(states) <= 1:
                return ConsistencyStatus.CONSISTENT
            
            # 检查状态一致性
            status_consistency = self._check_status_consistency(states)
            
            # 检查时间一致性
            time_consistency = self._check_time_consistency_detailed(states)
            
            # 检查资源使用一致性
            resource_consistency = self._check_resource_consistency_detailed(states)
            
            # 检查依赖关系一致性
            dependency_consistency = self._check_dependency_consistency(states)
            
            # 综合判断
            consistency_scores = [
                status_consistency,
                time_consistency,
                resource_consistency,
                dependency_consistency
            ]
            
            avg_consistency = statistics.mean(consistency_scores)
            
            if avg_consistency >= 0.8:
                return ConsistencyStatus.CONSISTENT
            elif avg_consistency >= 0.5:
                return ConsistencyStatus.INCONSISTENT
            elif avg_consistency >= 0.2:
                return ConsistencyStatus.CONFLICTING
            else:
                return ConsistencyStatus.UNKNOWN
                
        except Exception as e:
            self.logger.error(f"一致性检查失败: {e}")
            return ConsistencyStatus.UNKNOWN
    
    def prioritize_executions(self, states: List[ExecutionState]) -> List[Tuple[str, float]]:
        """执行优先级排序"""
        try:
            priority_scores = []
            
            for state in states:
                score = self._calculate_priority_score(state)
                priority_scores.append((f"{state.module_id}_{state.task_id}", score))
            
            # 按分数降序排序
            priority_scores.sort(key=lambda x: x[1], reverse=True)
            
            return priority_scores
            
        except Exception as e:
            self.logger.error(f"优先级排序失败: {e}")
            return []
    
    def get_state_history(self, module_id: Optional[str] = None, 
                         time_range: Optional[Tuple[datetime, datetime]] = None) -> List[StateHistory]:
        """获取状态历史记录"""
        with self.lock:
            filtered_history = []
            
            for history_entry in self.state_history:
                # 模块过滤
                if module_id and not history_entry.state.module_id.startswith(module_id):
                    continue
                
                # 时间范围过滤
                if time_range:
                    start_time, end_time = time_range
                    if not (start_time <= history_entry.timestamp <= end_time):
                        continue
                
                filtered_history.append(history_entry)
            
            return filtered_history
    
    def generate_report(self, report_id: Optional[str] = None) -> StateReport:
        """生成执行状态报告"""
        try:
            if not report_id:
                report_id = f"report_{int(time.time())}"
            
            # 生成报告数据
            summary = self._generate_summary()
            detailed_metrics = self._generate_detailed_metrics()
            consistency_analysis = self._generate_consistency_analysis()
            priority_ranking = self._generate_priority_ranking()
            alerts = self._generate_alerts_summary()
            recommendations = self._generate_recommendations()
            
            # 创建报告
            report = StateReport(
                report_id=report_id,
                generation_time=datetime.now(),
                summary=summary,
                detailed_metrics=detailed_metrics,
                consistency_analysis=consistency_analysis,
                priority_ranking=priority_ranking,
                alerts=alerts,
                recommendations=recommendations
            )
            
            # 存储报告
            self.reports[report_id] = report
            
            self.logger.info(f"成功生成报告: {report_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
            raise
    
    def start_monitoring(self, interval: int = 60):
        """启动状态监控"""
        if self.monitoring_active:
            self.logger.warning("监控已在运行中")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("启动状态监控")
    
    def stop_monitoring(self):
        """停止状态监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("停止状态监控")
    
    def get_alerts(self, alert_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取预警信息"""
        if alert_type:
            return [alert for alert in self.alerts if alert.get('type') == alert_type]
        return self.alerts.copy()
    
    def clear_alerts(self, alert_ids: Optional[List[str]] = None):
        """清除预警"""
        if alert_ids:
            self.alerts = [alert for alert in self.alerts if alert.get('id') not in alert_ids]
        else:
            self.alerts.clear()
    
    def export_data(self, filepath: str, format: str = "json"):
        """导出数据"""
        try:
            export_data = {
                'active_states': {k: self._serialize_state(v) for k, v in self.active_states.items()},
                'history': [self._serialize_history(h) for h in self.state_history],
                'reports': {k: self._serialize_report(v) for k, v in self.reports.items()},
                'alerts': self.alerts
            }
            
            if format.lower() == "json":
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            self.logger.info(f"数据导出成功: {filepath}")
            
        except Exception as e:
            self.logger.error(f"数据导出失败: {e}")
            raise
    
    def _validate_state(self, state: ExecutionState) -> bool:
        """验证执行状态"""
        required_fields = ['module_id', 'task_id', 'status', 'priority']
        return all(hasattr(state, field) for field in required_fields)
    
    def _check_alerts(self, state: ExecutionState):
        """检查预警条件"""
        alerts_to_generate = []
        
        # 失败率预警
        if state.status == ExecutionStatus.FAILED:
            alerts_to_generate.append({
                'id': f"fail_{state.module_id}_{int(time.time())}",
                'type': 'execution_failure',
                'severity': 'high',
                'message': f"模块 {state.module_id} 执行失败",
                'timestamp': datetime.now(),
                'module_id': state.module_id
            })
        
        # 高资源使用预警
        if state.metrics.cpu_usage > 90 or state.metrics.memory_usage > 90:
            alerts_to_generate.append({
                'id': f"resource_{state.module_id}_{int(time.time())}",
                'type': 'high_resource_usage',
                'severity': 'medium',
                'message': f"模块 {state.module_id} 资源使用过高",
                'timestamp': datetime.now(),
                'module_id': state.module_id
            })
        
        # 高风险预警
        if state.risk_level > self.alert_threshold:
            alerts_to_generate.append({
                'id': f"risk_{state.module_id}_{int(time.time())}",
                'type': 'high_risk',
                'severity': 'critical',
                'message': f"模块 {state.module_id} 风险等级过高",
                'timestamp': datetime.now(),
                'module_id': state.module_id
            })
        
        # 添加预警
        for alert in alerts_to_generate:
            self.alerts.append(alert)
            self.logger.warning(f"生成预警: {alert['message']}")
    
    def _record_history(self, state: ExecutionState):
        """记录状态历史"""
        history_entry = StateHistory(
            timestamp=datetime.now(),
            state=state,
            aggregated_metrics=state.metrics,
            consistency_status=self.check_consistency([state]),
            priority_score=self._calculate_priority_score(state)
        )
        
        self.state_history.append(history_entry)
    
    def _evaluate_time_efficiency(self, state: ExecutionState) -> float:
        """评估时间效率"""
        if state.metrics.execution_time <= 0:
            return 0.5
        
        # 基于执行时间计算效率分数
        # 这里使用简单的反比例关系，实际应用中可能需要更复杂的模型
        efficiency = 1.0 / (1.0 + state.metrics.execution_time / 60.0)  # 假设60秒为基准
        return min(1.0, efficiency)
    
    def _evaluate_resource_efficiency(self, state: ExecutionState) -> float:
        """评估资源效率"""
        cpu_efficiency = max(0, 1.0 - state.metrics.cpu_usage / 100.0)
        memory_efficiency = max(0, 1.0 - state.metrics.memory_usage / 100.0)
        return (cpu_efficiency + memory_efficiency) / 2.0
    
    def _evaluate_priority_alignment(self, state: ExecutionState) -> float:
        """评估优先级匹配度"""
        # 根据状态完成情况评估优先级匹配度
        if state.status == ExecutionStatus.SUCCESS:
            return 1.0
        elif state.status == ExecutionStatus.RUNNING:
            return 0.8
        elif state.status == ExecutionStatus.FAILED:
            return 0.2
        else:
            return 0.5
    
    def _calculate_overall_score(self, evaluation: Dict[str, float]) -> float:
        """计算综合评估分数"""
        weights = {
            'time_efficiency': 0.2,
            'resource_efficiency': 0.2,
            'success_rate': 0.25,
            'priority_alignment': 0.15,
            'consistency': 0.1,
            'risk_score': 0.1
        }
        
        total_score = sum(
            evaluation.get(metric, 0) * weight
            for metric, weight in weights.items()
        )
        
        return total_score
    
    def _check_status_consistency(self, states: List[ExecutionState]) -> float:
        """检查状态一致性"""
        status_counts = defaultdict(int)
        for state in states:
            status_counts[state.status] += 1
        
        max_count = max(status_counts.values())
        return max_count / len(states)
    
    def _check_time_consistency_detailed(self, states: List[ExecutionState]) -> float:
        """详细检查时间一致性"""
        execution_times = [s.metrics.execution_time for s in states if s.metrics.execution_time > 0]
        if not execution_times:
            return 0.5
        
        mean_time = statistics.mean(execution_times)
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        cv = std_dev / mean_time if mean_time > 0 else 0
        return max(0, 1 - cv)
    
    def _check_resource_consistency_detailed(self, states: List[ExecutionState]) -> float:
        """详细检查资源一致性"""
        cpu_usage = [s.metrics.cpu_usage for s in states]
        memory_usage = [s.metrics.memory_usage for s in states]
        
        cpu_consistency = self._calculate_usage_consistency(cpu_usage)
        memory_consistency = self._calculate_usage_consistency(memory_usage)
        
        return (cpu_consistency + memory_consistency) / 2
    
    def _calculate_usage_consistency(self, usage_list: List[float]) -> float:
        """计算使用量一致性"""
        if not usage_list:
            return 0.5
        
        mean_usage = statistics.mean(usage_list)
        if mean_usage == 0:
            return 1.0
        
        std_dev = statistics.stdev(usage_list) if len(usage_list) > 1 else 0
        cv = std_dev / mean_usage
        
        return max(0, 1 - cv)
    
    def _check_dependency_consistency(self, states: List[ExecutionState]) -> float:
        """检查依赖关系一致性"""
        # 简化实现：检查依赖关系是否满足
        dependency_satisfied = 0
        total_dependencies = 0
        
        for state in states:
            if state.dependencies:
                total_dependencies += len(state.dependencies)
                for dep in state.dependencies:
                    # 检查依赖是否完成
                    dep_state = self.active_states.get(dep)
                    if dep_state and dep_state.status == ExecutionStatus.SUCCESS:
                        dependency_satisfied += 1
        
        return dependency_satisfied / total_dependencies if total_dependencies > 0 else 1.0
    
    def _calculate_priority_score(self, state: ExecutionState) -> float:
        """计算优先级分数"""
        priority_weights = {
            PriorityLevel.CRITICAL: 100,
            PriorityLevel.HIGH: 80,
            PriorityLevel.MEDIUM: 60,
            PriorityLevel.LOW: 40,
            PriorityLevel.BACKGROUND: 20
        }
        
        base_score = priority_weights.get(state.priority, 0)
        
        # 根据执行状态调整分数
        status_multipliers = {
            ExecutionStatus.RUNNING: 1.2,
            ExecutionStatus.SUCCESS: 1.0,
            ExecutionStatus.FAILED: 0.5,
            ExecutionStatus.PENDING: 0.8,
            ExecutionStatus.PAUSED: 0.6
        }
        
        multiplier = status_multipliers.get(state.status, 1.0)
        
        # 根据风险等级调整
        risk_penalty = state.risk_level * 0.3
        
        return max(0, base_score * multiplier - risk_penalty)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成摘要信息"""
        total_states = len(self.active_states)
        successful_states = sum(
            1 for state in self.active_states.values() 
            if state.status == ExecutionStatus.SUCCESS
        )
        failed_states = sum(
            1 for state in self.active_states.values() 
            if state.status == ExecutionStatus.FAILED
        )
        running_states = sum(
            1 for state in self.active_states.values() 
            if state.status == ExecutionStatus.RUNNING
        )
        
        avg_success_rate = statistics.mean(
            [state.metrics.success_rate for state in self.active_states.values()]
        ) if self.active_states else 0
        
        avg_risk_level = statistics.mean(
            [state.risk_level for state in self.active_states.values()]
        ) if self.active_states else 0
        
        return {
            'total_active_states': total_states,
            'successful_states': successful_states,
            'failed_states': failed_states,
            'running_states': running_states,
            'success_rate': successful_states / total_states if total_states > 0 else 0,
            'average_success_rate': avg_success_rate,
            'average_risk_level': avg_risk_level,
            'total_modules': len(self.module_states),
            'active_alerts': len([a for a in self.alerts if a.get('severity') in ['high', 'critical']])
        }
    
    def _generate_detailed_metrics(self) -> Dict[str, ExecutionMetrics]:
        """生成详细指标"""
        detailed_metrics = {}
        
        for module_id, states in self.module_states.items():
            if states:
                latest_state = states[-1]
                detailed_metrics[module_id] = latest_state.metrics
        
        return detailed_metrics
    
    def _generate_consistency_analysis(self) -> Dict[str, Any]:
        """生成一致性分析"""
        all_states = list(self.active_states.values())
        
        if len(all_states) <= 1:
            return {
                'overall_consistency': 'N/A',
                'consistency_score': 1.0,
                'conflicts': [],
                'recommendations': ['需要更多数据进行分析']
            }
        
        consistency_status = self.check_consistency(all_states)
        
        # 查找冲突
        conflicts = []
        status_groups = defaultdict(list)
        for state in all_states:
            status_groups[state.status].append(state)
        
        if len(status_groups) > 2:  # 超过2种状态认为有冲突
            conflicts = [
                {
                    'type': 'status_conflict',
                    'states': [s.module_id for s in group],
                    'status': status
                }
                for status, group in status_groups.items()
            ]
        
        return {
            'overall_consistency': consistency_status.value,
            'consistency_score': self.fusion_algorithm._calculate_consistency_score(all_states),
            'conflicts': conflicts,
            'status_distribution': {k.value: len(v) for k, v in status_groups.items()},
            'recommendations': self._generate_consistency_recommendations(conflicts)
        }
    
    def _generate_priority_ranking(self) -> List[Tuple[str, float]]:
        """生成优先级排序"""
        states = list(self.active_states.values())
        return self.prioritize_executions(states)
    
    def _generate_alerts_summary(self) -> List[Dict[str, Any]]:
        """生成预警摘要"""
        return self.alerts.copy()
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于成功率生成建议
        if self.active_states:
            avg_success_rate = statistics.mean(
                [state.metrics.success_rate for state in self.active_states.values()]
            )
            if avg_success_rate < 0.7:
                recommendations.append("建议检查失败率较高的模块，优化错误处理机制")
        
        # 基于资源使用生成建议
        high_resource_modules = [
            module_id for module_id, states in self.module_states.items()
            if states and (states[-1].metrics.cpu_usage > 80 or states[-1].metrics.memory_usage > 80)
        ]
        if high_resource_modules:
            recommendations.append(f"模块 {', '.join(high_resource_modules)} 资源使用过高，建议优化性能")
        
        # 基于风险等级生成建议
        high_risk_states = [
            state for state in self.active_states.values()
            if state.risk_level > self.alert_threshold
        ]
        if high_risk_states:
            recommendations.append("存在高风险执行任务，建议加强监控和风险控制")
        
        if not recommendations:
            recommendations.append("系统运行状态良好，建议保持当前配置")
        
        return recommendations
    
    def _generate_consistency_recommendations(self, conflicts: List[Dict[str, Any]]) -> List[str]:
        """生成一致性建议"""
        recommendations = []
        
        if conflicts:
            recommendations.append("检测到执行状态冲突，建议检查模块间的依赖关系")
            recommendations.append("考虑实施更严格的状态同步机制")
        
        return recommendations
    
    def _monitoring_loop(self, interval: int):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 生成定期报告
                if len(self.state_history) % 10 == 0:  # 每10条记录生成一次报告
                    self.generate_report()
                
                # 清理过期的预警
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.alerts = [
                    alert for alert in self.alerts
                    if alert.get('timestamp', datetime.now()) > cutoff_time
                ]
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(interval)
    
    def _serialize_state(self, state: ExecutionState) -> Dict[str, Any]:
        """序列化执行状态"""
        return {
            'module_id': state.module_id,
            'task_id': state.task_id,
            'status': state.status.value,
            'priority': state.priority.value,
            'start_time': state.start_time.isoformat() if state.start_time else None,
            'end_time': state.end_time.isoformat() if state.end_time else None,
            'metrics': {
                'execution_time': state.metrics.execution_time,
                'cpu_usage': state.metrics.cpu_usage,
                'memory_usage': state.metrics.memory_usage,
                'success_rate': state.metrics.success_rate,
                'error_count': state.metrics.error_count,
                'throughput': state.metrics.throughput,
                'latency': state.metrics.latency,
                'resource_utilization': state.metrics.resource_utilization
            },
            'dependencies': state.dependencies,
            'output_data': state.output_data,
            'error_message': state.error_message,
            'retry_count': state.retry_count,
            'max_retries': state.max_retries,
            'consistency_score': state.consistency_score,
            'risk_level': state.risk_level
        }
    
    def _serialize_history(self, history: StateHistory) -> Dict[str, Any]:
        """序列化历史记录"""
        return {
            'timestamp': history.timestamp.isoformat(),
            'state': self._serialize_state(history.state),
            'aggregated_metrics': {
                'execution_time': history.aggregated_metrics.execution_time,
                'cpu_usage': history.aggregated_metrics.cpu_usage,
                'memory_usage': history.aggregated_metrics.memory_usage,
                'success_rate': history.aggregated_metrics.success_rate,
                'error_count': history.aggregated_metrics.error_count,
                'throughput': history.aggregated_metrics.throughput,
                'latency': history.aggregated_metrics.latency,
                'resource_utilization': history.aggregated_metrics.resource_utilization
            },
            'consistency_status': history.consistency_status.value,
            'priority_score': history.priority_score
        }
    
    def _serialize_report(self, report: StateReport) -> Dict[str, Any]:
        """序列化报告"""
        return {
            'report_id': report.report_id,
            'generation_time': report.generation_time.isoformat(),
            'summary': report.summary,
            'detailed_metrics': {
                k: {
                    'execution_time': v.execution_time,
                    'cpu_usage': v.cpu_usage,
                    'memory_usage': v.memory_usage,
                    'success_rate': v.success_rate,
                    'error_count': v.error_count,
                    'throughput': v.throughput,
                    'latency': v.latency,
                    'resource_utilization': v.resource_utilization
                } for k, v in report.detailed_metrics.items()
            },
            'consistency_analysis': report.consistency_analysis,
            'priority_ranking': report.priority_ranking,
            'alerts': report.alerts,
            'recommendations': report.recommendations
        }


# 使用示例和测试代码
def create_sample_states():
    """创建示例执行状态"""
    states = []
    
    # 创建示例状态1
    state1 = ExecutionState(
        module_id="module_A",
        task_id="task_001",
        status=ExecutionStatus.RUNNING,
        priority=PriorityLevel.HIGH,
        start_time=datetime.now() - timedelta(minutes=5),
        metrics=ExecutionMetrics(
            execution_time=300.0,
            cpu_usage=65.0,
            memory_usage=70.0,
            success_rate=0.85,
            error_count=2,
            throughput=100.0,
            latency=50.0,
            resource_utilization=75.0
        ),
        dependencies=["module_B_task_001"],
        retry_count=1,
        consistency_score=0.8,
        risk_level=0.3
    )
    states.append(state1)
    
    # 创建示例状态2
    state2 = ExecutionState(
        module_id="module_B",
        task_id="task_001",
        status=ExecutionStatus.SUCCESS,
        priority=PriorityLevel.MEDIUM,
        start_time=datetime.now() - timedelta(minutes=10),
        end_time=datetime.now() - timedelta(minutes=2),
        metrics=ExecutionMetrics(
            execution_time=480.0,
            cpu_usage=45.0,
            memory_usage=55.0,
            success_rate=0.95,
            error_count=1,
            throughput=120.0,
            latency=40.0,
            resource_utilization=60.0
        ),
        dependencies=[],
        retry_count=0,
        consistency_score=0.9,
        risk_level=0.1
    )
    states.append(state2)
    
    # 创建示例状态3
    state3 = ExecutionState(
        module_id="module_C",
        task_id="task_001",
        status=ExecutionStatus.FAILED,
        priority=PriorityLevel.CRITICAL,
        start_time=datetime.now() - timedelta(minutes=15),
        end_time=datetime.now() - timedelta(minutes=8),
        metrics=ExecutionMetrics(
            execution_time=420.0,
            cpu_usage=90.0,
            memory_usage=85.0,
            success_rate=0.3,
            error_count=5,
            throughput=30.0,
            latency=120.0,
            resource_utilization=95.0
        ),
        dependencies=["module_B_task_001"],
        retry_count=2,
        max_retries=3,
        consistency_score=0.4,
        risk_level=0.8,
        error_message="网络连接超时"
    )
    states.append(state3)
    
    return states


def main():
    """主函数 - 演示执行状态聚合器功能"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建聚合器实例
    aggregator = ExecutionStateAggregator(max_history_size=100, alert_threshold=0.7)
    
    print("=== G9执行状态聚合器演示 ===\n")
    
    # 1. 创建示例状态并注册
    print("1. 注册模块执行状态...")
    sample_states = create_sample_states()
    
    for state in sample_states:
        success = aggregator.register_module_state(state.module_id, state)
        print(f"   注册 {state.module_id}: {'成功' if success else '失败'}")
    
    print()
    
    # 2. 聚合模块状态
    print("2. 聚合模块状态...")
    module_ids = ["module_A", "module_B", "module_C"]
    fused_state = aggregator.aggregate_module_states(module_ids)
    
    if fused_state:
        print(f"   融合状态: {fused_state.status.value}")
        print(f"   融合优先级: {fused_state.priority.value}")
        print(f"   一致性分数: {fused_state.consistency_score:.2f}")
        print(f"   风险等级: {fused_state.risk_level:.2f}")
    else:
        print("   状态聚合失败")
    
    print()
    
    # 3. 评估执行状态
    print("3. 评估执行状态...")
    for state in sample_states:
        evaluation = aggregator.evaluate_execution_state(state)
        print(f"   {state.module_id} 评估结果:")
        for metric, score in evaluation.items():
            print(f"     {metric}: {score:.2f}")
    
    print()
    
    # 4. 检查一致性
    print("4. 检查执行一致性...")
    consistency_status = aggregator.check_consistency(sample_states)
    print(f"   一致性状态: {consistency_status.value}")
    
    print()
    
    # 5. 优先级排序
    print("5. 执行优先级排序...")
    priority_ranking = aggregator.prioritize_executions(sample_states)
    for i, (task_id, score) in enumerate(priority_ranking, 1):
        print(f"   {i}. {task_id}: {score:.1f}")
    
    print()
    
    # 6. 获取历史记录
    print("6. 获取状态历史记录...")
    history = aggregator.get_state_history()
    print(f"   历史记录数量: {len(history)}")
    
    print()
    
    # 7. 生成报告
    print("7. 生成执行状态报告...")
    try:
        report = aggregator.generate_report("demo_report")
        print(f"   报告ID: {report.report_id}")
        print(f"   摘要信息: {report.summary}")
        print(f"   建议: {report.recommendations}")
    except Exception as e:
        print(f"   报告生成失败: {e}")
    
    print()
    
    # 8. 获取预警信息
    print("8. 获取预警信息...")
    alerts = aggregator.get_alerts()
    print(f"   预警数量: {len(alerts)}")
    for alert in alerts:
        print(f"   - {alert['type']}: {alert['message']}")
    
    print()
    
    # 9. 启动监控（演示用）
    print("9. 启动状态监控...")
    aggregator.start_monitoring(interval=10)
    print("   监控已启动，10秒后停止...")
    
    # 等待一段时间
    time.sleep(12)
    
    # 停止监控
    aggregator.stop_monitoring()
    print("   监控已停止")
    
    print()
    
    # 10. 导出数据
    print("10. 导出数据...")
    try:
        aggregator.export_data("/tmp/execution_state_data.json", "json")
        print("   数据导出成功: /tmp/execution_state_data.json")
    except Exception as e:
        print(f"   数据导出失败: {e}")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()