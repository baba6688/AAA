"""
G7 执行监控器
实现执行过程的实时监控、性能评估、异常检测、效果分析、预警机制、报告生成和优化建议
"""

import time
import threading
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import statistics
import copy
import traceback
from concurrent.futures import ThreadPoolExecutor
import asyncio
import queue


class ExecutionStatus(Enum):
    """执行状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    SUSPENDED = "suspended"


class AlertLevel(Enum):
    """预警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """指标类型枚举"""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    RESOURCE = "resource"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class ExecutionContext:
    """执行上下文"""
    execution_id: str
    task_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    parameters: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PerformanceMetric:
    """性能指标"""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    metric_type: MetricType
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class Alert:
    """预警信息"""
    alert_id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    execution_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class ExecutionReport:
    """执行报告"""
    report_id: str
    execution_id: str
    generated_at: datetime
    summary: Dict[str, Any]
    performance_metrics: List[PerformanceMetric]
    alerts: List[Alert]
    recommendations: List[str]
    execution_duration: float
    success_rate: float
    quality_score: float


class RealTimeMonitor:
    """实时监控器"""
    
    def __init__(self, monitor_interval: float = 1.0):
        self.monitor_interval = monitor_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.callbacks: List[Callable] = []
        self.lock = threading.RLock()
    
    def start_monitoring(self):
        """开始监控"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logging.info("实时监控器已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logging.info("实时监控器已停止")
    
    def register_execution(self, context: ExecutionContext):
        """注册执行任务"""
        with self.lock:
            self.active_executions[context.execution_id] = context
            logging.info(f"注册执行任务: {context.execution_id}")
    
    def unregister_execution(self, execution_id: str):
        """注销执行任务"""
        with self.lock:
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
                logging.info(f"注销执行任务: {execution_id}")
    
    def add_metric(self, metric: PerformanceMetric):
        """添加指标"""
        with self.lock:
            self.metrics_buffer.append(metric)
            # 触发回调
            for callback in self.callbacks:
                try:
                    callback(metric)
                except Exception as e:
                    logging.error(f"监控回调执行失败: {e}")
    
    def register_callback(self, callback: Callable):
        """注册回调函数"""
        self.callbacks.append(callback)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                self._check_active_executions()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logging.error(f"监控循环异常: {e}")
                time.sleep(self.monitor_interval)
    
    def _check_active_executions(self):
        """检查活跃执行任务"""
        with self.lock:
            current_time = datetime.now()
            for execution_id, context in list(self.active_executions.items()):
                # 检查超时
                if context.status == ExecutionStatus.RUNNING:
                    elapsed = (current_time - context.start_time).total_seconds()
                    timeout = context.parameters.get('timeout', 3600)  # 默认1小时超时
                    if elapsed > timeout:
                        context.status = ExecutionStatus.TIMEOUT
                        context.end_time = current_time
                        self.unregister_execution(execution_id)
                        
                        alert = Alert(
                            alert_id=f"timeout_{execution_id}",
                            level=AlertLevel.ERROR,
                            message=f"执行任务 {execution_id} 超时",
                            timestamp=current_time,
                            execution_id=execution_id,
                            metric_name="execution_timeout",
                            current_value=elapsed,
                            threshold_value=timeout,
                            details={"execution_context": asdict(context)}
                        )
                        # 这里可以触发预警机制


class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(self):
        self.baseline_metrics: Dict[str, float] = {}
        self.performance_thresholds: Dict[str, Dict[str, float]] = {}
        self.evaluation_history: deque = deque(maxlen=1000)
    
    def set_baseline(self, metric_name: str, baseline_value: float):
        """设置基线指标"""
        self.baseline_metrics[metric_name] = baseline_value
    
    def set_thresholds(self, metric_name: str, thresholds: Dict[str, float]):
        """设置性能阈值"""
        self.performance_thresholds[metric_name] = thresholds
    
    def evaluate_performance(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """评估性能"""
        evaluation_result = {
            "overall_score": 0.0,
            "metric_scores": {},
            "performance_trends": {},
            "anomalies": [],
            "recommendations": []
        }
        
        total_score = 0.0
        valid_metrics = 0
        
        for metric in metrics:
            score = self._evaluate_single_metric(metric)
            evaluation_result["metric_scores"][metric.metric_name] = score
            
            if score >= 0:
                total_score += score
                valid_metrics += 1
            
            # 检测异常
            anomaly = self._detect_anomaly(metric)
            if anomaly:
                evaluation_result["anomalies"].append(anomaly)
            
            # 分析趋势
            trend = self._analyze_trend(metric.metric_name, metric.value)
            if trend:
                evaluation_result["performance_trends"][metric.metric_name] = trend
        
        if valid_metrics > 0:
            evaluation_result["overall_score"] = total_score / valid_metrics
        
        # 生成建议
        evaluation_result["recommendations"] = self._generate_recommendations(evaluation_result)
        
        # 保存评估历史
        self.evaluation_history.append({
            "timestamp": datetime.now(),
            "result": evaluation_result
        })
        
        return evaluation_result
    
    def _evaluate_single_metric(self, metric: PerformanceMetric) -> float:
        """评估单个指标"""
        metric_name = metric.metric_name
        value = metric.value
        
        # 基于基线评估
        if metric_name in self.baseline_metrics:
            baseline = self.baseline_metrics[metric_name]
            if baseline > 0:
                relative_performance = baseline / value if value > 0 else 0
                return min(100, max(0, relative_performance * 100))
        
        # 基于阈值评估
        if metric_name in self.performance_thresholds:
            thresholds = self.performance_thresholds[metric_name]
            excellent = thresholds.get("excellent", float('inf'))
            good = thresholds.get("good", float('inf'))
            acceptable = thresholds.get("acceptable", float('inf'))
            
            if value <= excellent:
                return 100.0
            elif value <= good:
                return 80.0
            elif value <= acceptable:
                return 60.0
            else:
                return 30.0
        
        # 默认评估（值越小越好）
        return 50.0
    
    def _detect_anomaly(self, metric: PerformanceMetric) -> Optional[Dict[str, Any]]:
        """检测异常"""
        metric_name = metric.metric_name
        value = metric.value
        
        # 检查阈值
        if metric_name in self.performance_thresholds:
            thresholds = self.performance_thresholds[metric_name]
            critical = thresholds.get("critical")
            if critical and value > critical:
                return {
                    "metric_name": metric_name,
                    "type": "threshold_exceeded",
                    "value": value,
                    "threshold": critical,
                    "severity": "high"
                }
        
        # 检查历史数据异常（简化版）
        recent_metrics = [m for m in self.evaluation_history 
                         if m["timestamp"] > datetime.now() - timedelta(hours=1)]
        if len(recent_metrics) > 10:
            # 这里可以实现更复杂的异常检测算法
            pass
        
        return None
    
    def _analyze_trend(self, metric_name: str, current_value: float) -> Optional[Dict[str, Any]]:
        """分析趋势"""
        # 简化版趋势分析
        return {
            "direction": "stable",
            "change_rate": 0.0,
            "confidence": 0.5
        }
    
    def _generate_recommendations(self, evaluation_result: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于整体评分
        overall_score = evaluation_result["overall_score"]
        if overall_score < 60:
            recommendations.append("整体性能偏低，建议进行系统优化")
        elif overall_score > 90:
            recommendations.append("性能表现优秀，保持当前状态")
        
        # 基于异常信息
        anomalies = evaluation_result.get("anomalies", [])
        for anomaly in anomalies:
            if anomaly["type"] == "threshold_exceeded":
                recommendations.append(f"指标 {anomaly['metric_name']} 超过阈值，建议检查相关配置")
        
        # 基于趋势分析
        trends = evaluation_result.get("performance_trends", {})
        for metric_name, trend in trends.items():
            if trend["direction"] == "declining":
                recommendations.append(f"指标 {metric_name} 呈下降趋势，建议关注相关资源")
        
        return recommendations


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.detection_rules: List[Dict[str, Any]] = []
        self.statistical_thresholds: Dict[str, Dict[str, float]] = {}
    
    def add_detection_rule(self, rule: Dict[str, Any]):
        """添加检测规则"""
        self.detection_rules.append(rule)
    
    def set_statistical_threshold(self, metric_name: str, threshold_config: Dict[str, float]):
        """设置统计阈值"""
        self.statistical_thresholds[metric_name] = threshold_config
    
    def detect_anomalies(self, metric: PerformanceMetric) -> List[Dict[str, Any]]:
        """检测异常"""
        anomalies = []
        
        # 统计异常检测
        statistical_anomaly = self._statistical_detection(metric)
        if statistical_anomaly:
            anomalies.append(statistical_anomaly)
        
        # 规则异常检测
        rule_anomalies = self._rule_based_detection(metric)
        anomalies.extend(rule_anomalies)
        
        # 时间序列异常检测
        timeseries_anomaly = self._timeseries_detection(metric)
        if timeseries_anomaly:
            anomalies.append(timeseries_anomaly)
        
        return anomalies
    
    def _statistical_detection(self, metric: PerformanceMetric) -> Optional[Dict[str, Any]]:
        """统计异常检测"""
        metric_name = metric.metric_name
        value = metric.value
        
        if metric_name not in self.statistical_thresholds:
            return None
        
        threshold_config = self.statistical_thresholds[metric_name]
        
        # 获取历史数据
        history = list(self.metric_history[metric_name])
        if len(history) < 10:  # 需要足够的历史数据
            return None
        
        values = [m.value for m in history]
        
        # 计算统计指标
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        # Z-score检测
        if std_val > 0:
            z_score = abs((value - mean_val) / std_val)
            threshold = threshold_config.get("z_score_threshold", 3.0)
            if z_score > threshold:
                return {
                    "type": "statistical_outlier",
                    "metric_name": metric_name,
                    "value": value,
                    "z_score": z_score,
                    "threshold": threshold,
                    "mean": mean_val,
                    "std": std_val,
                    "severity": "high" if z_score > threshold * 1.5 else "medium"
                }
        
        # 百分位数检测
        p95 = threshold_config.get("p95")
        p99 = threshold_config.get("p99")
        if p95 and value > p95:
            return {
                "type": "percentile_exceeded",
                "metric_name": metric_name,
                "value": value,
                "percentile": 95,
                "threshold": p95,
                "severity": "medium"
            }
        if p99 and value > p99:
            return {
                "type": "percentile_exceeded",
                "metric_name": metric_name,
                "value": value,
                "percentile": 99,
                "threshold": p99,
                "severity": "high"
            }
        
        return None
    
    def _rule_based_detection(self, metric: PerformanceMetric) -> List[Dict[str, Any]]:
        """基于规则的异常检测"""
        anomalies = []
        
        for rule in self.detection_rules:
            try:
                if self._evaluate_rule(rule, metric):
                    anomalies.append({
                        "type": "rule_violation",
                        "metric_name": metric.metric_name,
                        "rule": rule,
                        "value": metric.value,
                        "severity": rule.get("severity", "medium")
                    })
            except Exception as e:
                logging.error(f"规则评估失败: {e}")
        
        return anomalies
    
    def _evaluate_rule(self, rule: Dict[str, Any], metric: PerformanceMetric) -> bool:
        """评估单个规则"""
        rule_type = rule.get("type")
        metric_name = rule.get("metric_name")
        condition = rule.get("condition")
        
        if metric_name != metric.metric_name:
            return False
        
        if rule_type == "threshold":
            operator = condition.get("operator")
            threshold = condition.get("value")
            value = metric.value
            
            if operator == ">":
                return value > threshold
            elif operator == "<":
                return value < threshold
            elif operator == ">=":
                return value >= threshold
            elif operator == "<=":
                return value <= threshold
            elif operator == "==":
                return value == threshold
            elif operator == "!=":
                return value != threshold
        
        return False
    
    def _timeseries_detection(self, metric: PerformanceMetric) -> Optional[Dict[str, Any]]:
        """时间序列异常检测"""
        metric_name = metric.metric_name
        value = metric.value
        
        # 更新历史数据
        self.metric_history[metric_name].append(metric)
        
        # 简单的趋势异常检测
        history = list(self.metric_history[metric_name])
        if len(history) < 5:
            return None
        
        # 检查连续增长或下降
        recent_values = [m.value for m in history[-5:]]
        
        # 连续增长检测
        if all(recent_values[i] > recent_values[i-1] for i in range(1, len(recent_values))):
            growth_rate = (recent_values[-1] - recent_values[0]) / recent_values[0] if recent_values[0] > 0 else 0
            if growth_rate > 0.5:  # 50%增长
                return {
                    "type": "rapid_growth",
                    "metric_name": metric_name,
                    "value": value,
                    "growth_rate": growth_rate,
                    "severity": "high"
                }
        
        # 连续下降检测
        if all(recent_values[i] < recent_values[i-1] for i in range(1, len(recent_values))):
            decline_rate = (recent_values[0] - recent_values[-1]) / recent_values[0] if recent_values[0] > 0 else 0
            if decline_rate > 0.3:  # 30%下降
                return {
                    "type": "rapid_decline",
                    "metric_name": metric_name,
                    "value": value,
                    "decline_rate": decline_rate,
                    "severity": "medium"
                }
        
        return None


class EffectAnalyzer:
    """执行效果分析器"""
    
    def __init__(self):
        self.analysis_templates: Dict[str, Dict[str, Any]] = {}
        self.correlation_cache: Dict[str, float] = {}
    
    def analyze_execution_effect(self, execution_context: ExecutionContext, 
                                metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """分析执行效果"""
        analysis_result = {
            "execution_id": execution_context.execution_id,
            "task_name": execution_context.task_name,
            "analysis_timestamp": datetime.now(),
            "overall_effectiveness": 0.0,
            "efficiency_score": 0.0,
            "quality_score": 0.0,
            "resource_utilization": 0.0,
            "success_factors": [],
            "improvement_areas": [],
            "detailed_metrics": {},
            "comparative_analysis": {}
        }
        
        # 计算各种评分
        analysis_result["efficiency_score"] = self._calculate_efficiency_score(execution_context, metrics)
        analysis_result["quality_score"] = self._calculate_quality_score(metrics)
        analysis_result["resource_utilization"] = self._calculate_resource_utilization(metrics)
        
        # 综合效果评分
        analysis_result["overall_effectiveness"] = (
            analysis_result["efficiency_score"] * 0.3 +
            analysis_result["quality_score"] * 0.4 +
            analysis_result["resource_utilization"] * 0.3
        )
        
        # 详细指标分析
        analysis_result["detailed_metrics"] = self._analyze_detailed_metrics(metrics)
        
        # 成功因素分析
        analysis_result["success_factors"] = self._identify_success_factors(execution_context, metrics)
        
        # 改进领域识别
        analysis_result["improvement_areas"] = self._identify_improvement_areas(analysis_result)
        
        # 比较分析
        analysis_result["comparative_analysis"] = self._comparative_analysis(execution_context, metrics)
        
        return analysis_result
    
    def _calculate_efficiency_score(self, context: ExecutionContext, metrics: List[PerformanceMetric]) -> float:
        """计算效率评分"""
        if context.end_time and context.start_time:
            duration = (context.end_time - context.start_time).total_seconds()
            # 基于执行时间计算效率（时间越短效率越高）
            efficiency = max(0, 100 - (duration / 60))  # 假设1分钟为基准
            return min(100, efficiency)
        return 50.0
    
    def _calculate_quality_score(self, metrics: List[PerformanceMetric]) -> float:
        """计算质量评分"""
        quality_metrics = [m for m in metrics if m.metric_type == MetricType.QUALITY]
        if not quality_metrics:
            return 50.0
        
        total_score = 0.0
        for metric in quality_metrics:
            # 简化评分逻辑
            if "accuracy" in metric.metric_name.lower():
                score = min(100, metric.value * 100)
            elif "error" in metric.metric_name.lower():
                score = max(0, 100 - metric.value)
            else:
                score = 70.0  # 默认分数
            total_score += score
        
        return total_score / len(quality_metrics)
    
    def _calculate_resource_utilization(self, metrics: List[PerformanceMetric]) -> float:
        """计算资源利用率"""
        resource_metrics = [m for m in metrics if m.metric_type == MetricType.RESOURCE]
        if not resource_metrics:
            return 50.0
        
        utilization_scores = []
        for metric in resource_metrics:
            if "cpu" in metric.metric_name.lower():
                # CPU利用率在70-85%之间为最佳
                if 70 <= metric.value <= 85:
                    score = 100
                elif metric.value < 70:
                    score = metric.value / 70 * 80
                else:
                    score = max(0, 100 - (metric.value - 85) * 2)
                utilization_scores.append(score)
            elif "memory" in metric.metric_name.lower():
                # 内存利用率在60-80%之间为最佳
                if 60 <= metric.value <= 80:
                    score = 100
                elif metric.value < 60:
                    score = metric.value / 60 * 80
                else:
                    score = max(0, 100 - (metric.value - 80) * 1.5)
                utilization_scores.append(score)
            else:
                utilization_scores.append(70.0)
        
        return sum(utilization_scores) / len(utilization_scores) if utilization_scores else 50.0
    
    def _analyze_detailed_metrics(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """详细指标分析"""
        detailed_analysis = {}
        
        for metric in metrics:
            metric_analysis = {
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp,
                "type": metric.metric_type.value,
                "performance_rating": self._rate_metric_performance(metric)
            }
            detailed_analysis[metric.metric_name] = metric_analysis
        
        return detailed_analysis
    
    def _rate_metric_performance(self, metric: PerformanceMetric) -> str:
        """评级指标性能"""
        value = metric.value
        
        # 简化的评级逻辑
        if metric.metric_type == MetricType.PERFORMANCE:
            if value < 1.0:
                return "excellent"
            elif value < 5.0:
                return "good"
            elif value < 10.0:
                return "acceptable"
            else:
                return "poor"
        elif metric.metric_type == MetricType.ERROR:
            if value == 0:
                return "excellent"
            elif value < 0.01:
                return "good"
            elif value < 0.05:
                return "acceptable"
            else:
                return "poor"
        else:
            return "unknown"
    
    def _identify_success_factors(self, context: ExecutionContext, metrics: List[PerformanceMetric]) -> List[str]:
        """识别成功因素"""
        success_factors = []
        
        # 基于执行状态
        if context.status == ExecutionStatus.COMPLETED:
            success_factors.append("任务成功完成")
        
        # 基于性能指标
        performance_metrics = [m for m in metrics if m.metric_type == MetricType.PERFORMANCE]
        for metric in performance_metrics:
            if metric.value < 5.0:  # 性能良好
                success_factors.append(f"{metric.metric_name} 性能表现良好")
        
        # 基于资源利用
        resource_metrics = [m for m in metrics if m.metric_type == MetricType.RESOURCE]
        for metric in resource_metrics:
            if "cpu" in metric.metric_name.lower() and 70 <= metric.value <= 85:
                success_factors.append("CPU利用率优化良好")
            if "memory" in metric.metric_name.lower() and 60 <= metric.value <= 80:
                success_factors.append("内存利用率优化良好")
        
        return success_factors
    
    def _identify_improvement_areas(self, analysis_result: Dict[str, Any]) -> List[str]:
        """识别改进领域"""
        improvement_areas = []
        
        efficiency_score = analysis_result.get("efficiency_score", 0)
        quality_score = analysis_result.get("quality_score", 0)
        resource_utilization = analysis_result.get("resource_utilization", 0)
        
        if efficiency_score < 60:
            improvement_areas.append("执行效率有待提升")
        if quality_score < 70:
            improvement_areas.append("执行质量需要改进")
        if resource_utilization < 50:
            improvement_areas.append("资源利用率偏低")
        
        return improvement_areas
    
    def _comparative_analysis(self, context: ExecutionContext, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """比较分析"""
        # 简化版比较分析
        return {
            "vs_baseline": "需要更多历史数据进行比较",
            "vs_previous": "需要更多执行记录进行比较",
            "trend": "stable"
        }


class AlertManager:
    """预警管理器"""
    
    def __init__(self):
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.notification_callbacks: List[Callable] = []
        self.alert_suppression: Dict[str, datetime] = {}
    
    def add_alert_rule(self, rule_name: str, rule_config: Dict[str, Any]):
        """添加预警规则"""
        self.alert_rules[rule_name] = rule_config
    
    def register_notification_callback(self, callback: Callable):
        """注册通知回调"""
        self.notification_callbacks.append(callback)
    
    def check_and_trigger_alerts(self, metric: PerformanceMetric) -> List[Alert]:
        """检查并触发预警"""
        triggered_alerts = []
        
        for rule_name, rule_config in self.alert_rules.items():
            alert = self._evaluate_rule(rule_name, rule_config, metric)
            if alert:
                # 检查预警抑制
                if self._should_suppress_alert(alert):
                    continue
                
                triggered_alerts.append(alert)
                self._store_alert(alert)
                
                # 触发通知
                self._notify_alert(alert)
        
        return triggered_alerts
    
    def _evaluate_rule(self, rule_name: str, rule_config: Dict[str, Any], 
                      metric: PerformanceMetric) -> Optional[Alert]:
        """评估预警规则"""
        metric_name = rule_config.get("metric_name")
        condition = rule_config.get("condition")
        severity = rule_config.get("severity", AlertLevel.WARNING)
        
        if metric_name != metric.metric_name:
            return None
        
        # 评估条件
        if self._evaluate_condition(condition, metric.value):
            return Alert(
                alert_id=f"{rule_name}_{metric.metric_name}_{int(time.time())}",
                level=severity,
                message=rule_config.get("message", f"指标 {metric.metric_name} 触发预警"),
                timestamp=datetime.now(),
                execution_id=rule_config.get("execution_id", ""),
                metric_name=metric.metric_name,
                current_value=metric.value,
                threshold_value=condition.get("value", 0),
                details=rule_config.get("details", {})
            )
        
        return None
    
    def _evaluate_condition(self, condition: Dict[str, Any], value: float) -> bool:
        """评估预警条件"""
        operator = condition.get("operator")
        threshold = condition.get("value")
        
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        
        return False
    
    def _should_suppress_alert(self, alert: Alert) -> bool:
        """检查是否应该抑制预警"""
        suppression_key = f"{alert.metric_name}_{alert.level.value}"
        
        if suppression_key in self.alert_suppression:
            last_alert_time = self.alert_suppression[suppression_key]
            # 5分钟内不重复预警
            if datetime.now() - last_alert_time < timedelta(minutes=5):
                return True
        
        return False
    
    def _store_alert(self, alert: Alert):
        """存储预警"""
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.alert_suppression[f"{alert.metric_name}_{alert.level.value}"] = alert.timestamp
    
    def _notify_alert(self, alert: Alert):
        """通知预警"""
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logging.error(f"预警通知失败: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃预警"""
        return list(self.active_alerts.values())
    
    def resolve_alert(self, alert_id: str):
        """解决预警"""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        self.report_templates: Dict[str, str] = {}
        self.report_config: Dict[str, Any] = {}
    
    def generate_execution_report(self, execution_context: ExecutionContext,
                                 metrics: List[PerformanceMetric],
                                 analysis_result: Dict[str, Any],
                                 alerts: List[Alert]) -> ExecutionReport:
        """生成执行报告"""
        
        # 计算执行持续时间
        duration = 0.0
        if execution_context.end_time and execution_context.start_time:
            duration = (execution_context.end_time - execution_context.start_time).total_seconds()
        
        # 计算成功率
        success_rate = 100.0 if execution_context.status == ExecutionStatus.COMPLETED else 0.0
        
        # 生成建议
        recommendations = self._generate_recommendations(analysis_result, alerts)
        
        report = ExecutionReport(
            report_id=f"report_{execution_context.execution_id}_{int(time.time())}",
            execution_id=execution_context.execution_id,
            generated_at=datetime.now(),
            summary={
                "task_name": execution_context.task_name,
                "execution_status": execution_context.status.value,
                "start_time": execution_context.start_time,
                "end_time": execution_context.end_time,
                "duration": duration,
                "success_rate": success_rate,
                "overall_effectiveness": analysis_result.get("overall_effectiveness", 0.0)
            },
            performance_metrics=metrics,
            alerts=alerts,
            recommendations=recommendations,
            execution_duration=duration,
            success_rate=success_rate,
            quality_score=analysis_result.get("quality_score", 0.0)
        )
        
        return report
    
    def _generate_recommendations(self, analysis_result: Dict[str, Any], 
                                alerts: List[Alert]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于效果分析的改进建议
        improvement_areas = analysis_result.get("improvement_areas", [])
        recommendations.extend(improvement_areas)
        
        # 基于预警的建议
        for alert in alerts:
            if alert.level == AlertLevel.CRITICAL:
                recommendations.append(f"严重预警: {alert.message}，需要立即处理")
            elif alert.level == AlertLevel.ERROR:
                recommendations.append(f"错误预警: {alert.message}，建议尽快处理")
            elif alert.level == AlertLevel.WARNING:
                recommendations.append(f"警告: {alert.message}，建议关注")
        
        # 基于性能指标的建议
        efficiency_score = analysis_result.get("efficiency_score", 0)
        if efficiency_score < 50:
            recommendations.append("执行效率较低，建议优化算法或资源配置")
        elif efficiency_score > 90:
            recommendations.append("执行效率优秀，可以考虑推广最佳实践")
        
        # 基于质量指标的建议
        quality_score = analysis_result.get("quality_score", 0)
        if quality_score < 70:
            recommendations.append("执行质量有待提升，建议加强质量控制")
        
        return recommendations
    
    def export_report(self, report: ExecutionReport, format_type: str = "json") -> str:
        """导出报告"""
        if format_type == "json":
            return json.dumps(asdict(report), indent=2, ensure_ascii=False, default=str)
        elif format_type == "dict":
            return asdict(report)
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")


class OptimizationAdvisor:
    """优化建议器"""
    
    def __init__(self):
        self.optimization_strategies: Dict[str, Dict[str, Any]] = {}
        self.performance_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def add_optimization_strategy(self, strategy_name: str, strategy_config: Dict[str, Any]):
        """添加优化策略"""
        self.optimization_strategies[strategy_name] = strategy_config
    
    def generate_optimization_suggestions(self, execution_context: ExecutionContext,
                                        metrics: List[PerformanceMetric],
                                        analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成优化建议"""
        suggestions = []
        
        # 基于性能分析的建议
        performance_suggestions = self._analyze_performance_optimization(metrics, analysis_result)
        suggestions.extend(performance_suggestions)
        
        # 基于资源利用的建议
        resource_suggestions = self._analyze_resource_optimization(metrics, analysis_result)
        suggestions.extend(resource_suggestions)
        
        # 基于质量分析的建议
        quality_suggestions = self._analyze_quality_optimization(metrics, analysis_result)
        suggestions.extend(quality_suggestions)
        
        # 基于历史模式的建议
        pattern_suggestions = self._analyze_pattern_based_optimization(execution_context, metrics)
        suggestions.extend(pattern_suggestions)
        
        return suggestions
    
    def _analyze_performance_optimization(self, metrics: List[PerformanceMetric],
                                        analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析性能优化"""
        suggestions = []
        
        efficiency_score = analysis_result.get("efficiency_score", 0)
        
        if efficiency_score < 60:
            suggestions.append({
                "category": "performance",
                "priority": "high",
                "title": "提升执行效率",
                "description": "当前执行效率偏低，建议优化算法实现或增加并行处理",
                "actions": [
                    "检查算法复杂度，考虑优化关键路径",
                    "增加并行处理能力",
                    "优化数据结构选择",
                    "减少不必要的计算步骤"
                ],
                "expected_improvement": "20-40%效率提升"
            })
        
        # 分析具体的性能瓶颈
        performance_metrics = [m for m in metrics if m.metric_type == MetricType.PERFORMANCE]
        for metric in performance_metrics:
            if metric.value > 10.0:  # 性能较差
                suggestions.append({
                    "category": "performance",
                    "priority": "medium",
                    "title": f"优化{metric.metric_name}",
                    "description": f"{metric.metric_name}性能指标为{metric.value}{metric.unit}，建议优化",
                    "actions": [f"针对{metric.metric_name}进行专项优化"],
                    "expected_improvement": "10-20%性能提升"
                })
        
        return suggestions
    
    def _analyze_resource_optimization(self, metrics: List[PerformanceMetric],
                                     analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析资源优化"""
        suggestions = []
        
        resource_utilization = analysis_result.get("resource_utilization", 0)
        
        if resource_utilization < 50:
            suggestions.append({
                "category": "resource",
                "priority": "medium",
                "title": "提高资源利用率",
                "description": "当前资源利用率偏低，建议优化资源配置",
                "actions": [
                    "调整CPU分配策略",
                    "优化内存使用模式",
                    "平衡负载分布",
                    "考虑资源池化"
                ],
                "expected_improvement": "15-30%资源利用率提升"
            })
        
        # 分析具体的资源指标
        resource_metrics = [m for m in metrics if m.metric_type == MetricType.RESOURCE]
        for metric in resource_metrics:
            if "cpu" in metric.metric_name.lower():
                if metric.value < 50:
                    suggestions.append({
                        "category": "resource",
                        "priority": "low",
                        "title": "提高CPU利用率",
                        "description": f"CPU利用率为{metric.value}%，可以增加计算负载",
                        "actions": ["增加并行任务", "优化负载均衡"],
                        "expected_improvement": "10-20%CPU利用率提升"
                    })
                elif metric.value > 90:
                    suggestions.append({
                        "category": "resource",
                        "priority": "high",
                        "title": "降低CPU负载",
                        "description": f"CPU利用率为{metric.value}%，存在过载风险",
                        "actions": ["减少并发任务", "优化算法复杂度", "增加CPU资源"],
                        "expected_improvement": "避免系统过载"
                    })
            
            elif "memory" in metric.metric_name.lower():
                if metric.value > 85:
                    suggestions.append({
                        "category": "resource",
                        "priority": "high",
                        "title": "优化内存使用",
                        "description": f"内存使用率为{metric.value}%，存在内存不足风险",
                        "actions": ["优化内存分配", "增加内存容量", "实现内存回收"],
                        "expected_improvement": "避免内存溢出"
                    })
        
        return suggestions
    
    def _analyze_quality_optimization(self, metrics: List[PerformanceMetric],
                                    analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析质量优化"""
        suggestions = []
        
        quality_score = analysis_result.get("quality_score", 0)
        
        if quality_score < 70:
            suggestions.append({
                "category": "quality",
                "priority": "high",
                "title": "提升执行质量",
                "description": "当前执行质量偏低，建议加强质量控制措施",
                "actions": [
                    "增加输入验证",
                    "完善错误处理机制",
                    "实施质量检查点",
                    "加强测试覆盖"
                ],
                "expected_improvement": "20-40%质量提升"
            })
        
        # 分析质量指标
        quality_metrics = [m for m in metrics if m.metric_type == MetricType.QUALITY]
        for metric in quality_metrics:
            if "error" in metric.metric_name.lower() and metric.value > 0.01:
                suggestions.append({
                    "category": "quality",
                    "priority": "high",
                    "title": "降低错误率",
                    "description": f"错误率为{metric.value}%，需要加强错误控制",
                    "actions": ["加强输入验证", "完善异常处理", "增加重试机制"],
                    "expected_improvement": "50-80%错误率降低"
                })
        
        return suggestions
    
    def _analyze_pattern_based_optimization(self, execution_context: ExecutionContext,
                                          metrics: List[PerformanceMetric]) -> List[Dict[str, Any]]:
        """基于模式的优化分析"""
        suggestions = []
        
        # 记录当前执行模式
        pattern = {
            "timestamp": datetime.now(),
            "task_name": execution_context.task_name,
            "status": execution_context.status.value,
            "metrics": {m.metric_name: m.value for m in metrics}
        }
        self.performance_patterns[execution_context.task_name].append(pattern)
        
        # 保持最近100个模式记录
        if len(self.performance_patterns[execution_context.task_name]) > 100:
            self.performance_patterns[execution_context.task_name] = \
                self.performance_patterns[execution_context.task_name][-100:]
        
        # 基于历史模式分析
        patterns = self.performance_patterns[execution_context.task_name]
        if len(patterns) >= 10:
            suggestions.extend(self._analyze_historical_patterns(patterns, execution_context.task_name))
        
        return suggestions
    
    def _analyze_historical_patterns(self, patterns: List[Dict[str, Any]], 
                                   task_name: str) -> List[Dict[str, Any]]:
        """分析历史模式"""
        suggestions = []
        
        # 分析成功率趋势
        successful_runs = sum(1 for p in patterns[-10:] if p["status"] == "completed")
        success_rate = successful_runs / min(10, len(patterns))
        
        if success_rate < 0.8:
            suggestions.append({
                "category": "pattern",
                "priority": "high",
                "title": "提高任务成功率",
                "description": f"任务{task_name}最近成功率仅为{success_rate:.1%}，需要分析失败原因",
                "actions": [
                    "分析失败案例",
                    "加强输入验证",
                    "优化异常处理",
                    "增加重试机制"
                ],
                "expected_improvement": "提升任务稳定性"
            })
        
        # 分析性能趋势
        if len(patterns) >= 5:
            recent_performance = [p["metrics"].get("execution_time", 0) for p in patterns[-5:]]
            if recent_performance and all(recent_performance):
                if recent_performance[-1] > max(recent_performance[:-1]) * 1.2:
                    suggestions.append({
                        "category": "pattern",
                        "priority": "medium",
                        "title": "性能下降预警",
                        "description": f"任务{task_name}最近执行时间明显增加",
                        "actions": [
                            "检查系统负载",
                            "分析资源使用情况",
                            "优化算法实现"
                        ],
                        "expected_improvement": "恢复性能水平"
                    })
        
        return suggestions


class ExecutionMonitor:
    """G7执行监控器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 初始化各个组件
        self.real_time_monitor = RealTimeMonitor(
            monitor_interval=self.config.get("monitor_interval", 1.0)
        )
        self.performance_evaluator = PerformanceEvaluator()
        self.anomaly_detector = AnomalyDetector()
        self.effect_analyzer = EffectAnalyzer()
        self.alert_manager = AlertManager()
        self.report_generator = ReportGenerator()
        self.optimization_advisor = OptimizationAdvisor()
        
        # 状态管理
        self.is_initialized = False
        self.is_monitoring = False
        self.execution_cache: Dict[str, ExecutionContext] = {}
        self.metrics_cache: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self._setup_default_config()
        self._register_default_callbacks()
    
    def _setup_default_config(self):
        """设置默认配置"""
        # 设置默认性能阈值
        default_thresholds = {
            "execution_time": {"excellent": 5.0, "good": 10.0, "acceptable": 30.0, "critical": 60.0},
            "cpu_usage": {"excellent": 70.0, "good": 85.0, "acceptable": 95.0, "critical": 99.0},
            "memory_usage": {"excellent": 60.0, "good": 80.0, "acceptable": 90.0, "critical": 95.0},
            "error_rate": {"excellent": 0.0, "good": 0.01, "acceptable": 0.05, "critical": 0.1}
        }
        
        for metric_name, thresholds in default_thresholds.items():
            self.performance_evaluator.set_thresholds(metric_name, thresholds)
        
        # 设置默认异常检测规则
        self.anomaly_detector.set_statistical_threshold("execution_time", {
            "z_score_threshold": 3.0,
            "p95": 30.0,
            "p99": 60.0
        })
        
        self.anomaly_detector.set_statistical_threshold("cpu_usage", {
            "z_score_threshold": 2.5,
            "p95": 90.0,
            "p99": 95.0
        })
        
        # 设置默认预警规则
        self.alert_manager.add_alert_rule("high_execution_time", {
            "metric_name": "execution_time",
            "condition": {"operator": ">", "value": 30.0},
            "severity": AlertLevel.WARNING,
            "message": "执行时间过长",
            "execution_id": ""
        })
        
        self.alert_manager.add_alert_rule("critical_cpu_usage", {
            "metric_name": "cpu_usage",
            "condition": {"operator": ">", "value": 95.0},
            "severity": AlertLevel.CRITICAL,
            "message": "CPU使用率过高",
            "execution_id": ""
        })
        
        self.alert_manager.add_alert_rule("high_error_rate", {
            "metric_name": "error_rate",
            "condition": {"operator": ">", "value": 0.05},
            "severity": AlertLevel.ERROR,
            "message": "错误率过高",
            "execution_id": ""
        })
    
    def _register_default_callbacks(self):
        """注册默认回调函数"""
        # 指标监控回调
        def metric_monitor_callback(metric: PerformanceMetric):
            # 异常检测
            anomalies = self.anomaly_detector.detect_anomalies(metric)
            for anomaly in anomalies:
                logging.warning(f"检测到异常: {anomaly}")
            
            # 预警检查
            alerts = self.alert_manager.check_and_trigger_alerts(metric)
            for alert in alerts:
                logging.warning(f"触发预警: {alert.message}")
        
        self.real_time_monitor.register_callback(metric_monitor_callback)
        
        # 预警通知回调
        def alert_notification_callback(alert: Alert):
            print(f"[{alert.level.value.upper()}] {alert.message}")
            # 这里可以集成邮件、短信、钉钉等通知方式
        
        self.alert_manager.register_notification_callback(alert_notification_callback)
    
    def initialize(self):
        """初始化监控器"""
        if not self.is_initialized:
            self.real_time_monitor.start_monitoring()
            self.is_initialized = True
            self.is_monitoring = True
            logging.info("G7执行监控器初始化完成")
    
    def shutdown(self):
        """关闭监控器"""
        if self.is_monitoring:
            self.real_time_monitor.stop_monitoring()
            self.is_monitoring = False
            self.is_initialized = False
            self.executor.shutdown(wait=True)
            logging.info("G7执行监控器已关闭")
    
    def start_execution_monitoring(self, execution_context: ExecutionContext):
        """开始执行监控"""
        if not self.is_initialized:
            self.initialize()
        
        # 注册执行任务
        self.real_time_monitor.register_execution(execution_context)
        self.execution_cache[execution_context.execution_id] = execution_context
        
        logging.info(f"开始监控执行任务: {execution_context.execution_id}")
    
    def stop_execution_monitoring(self, execution_id: str):
        """停止执行监控"""
        # 注销执行任务
        self.real_time_monitor.unregister_execution(execution_id)
        
        if execution_id in self.execution_cache:
            context = self.execution_cache[execution_id]
            if not context.end_time:
                context.end_time = datetime.now()
            self.execution_cache[execution_id] = context
        
        logging.info(f"停止监控执行任务: {execution_id}")
    
    def record_metric(self, execution_id: str, metric: PerformanceMetric):
        """记录指标"""
        # 添加到缓存
        self.metrics_cache[execution_id].append(metric)
        
        # 添加到实时监控
        self.real_time_monitor.add_metric(metric)
        
        logging.debug(f"记录指标: {metric.metric_name} = {metric.value} {metric.unit}")
    
    def update_execution_status(self, execution_id: str, status: ExecutionStatus):
        """更新执行状态"""
        if execution_id in self.execution_cache:
            context = self.execution_cache[execution_id]
            context.status = status
            
            if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, 
                         ExecutionStatus.CANCELLED, ExecutionStatus.TIMEOUT]:
                context.end_time = datetime.now()
                # 触发完整的分析流程
                self._trigger_comprehensive_analysis(execution_id)
            
            self.execution_cache[execution_id] = context
    
    def _trigger_comprehensive_analysis(self, execution_id: str):
        """触发综合分析"""
        if execution_id not in self.execution_cache:
            return
        
        context = self.execution_cache[execution_id]
        metrics = self.metrics_cache.get(execution_id, [])
        
        # 异步执行分析
        future = self.executor.submit(self._perform_comprehensive_analysis, context, metrics)
        
        # 可以选择等待结果或继续异步处理
        try:
            analysis_result = future.result(timeout=30)  # 30秒超时
            logging.info(f"执行 {execution_id} 综合分析完成")
            
            # 生成报告
            alerts = self.alert_manager.get_active_alerts()
            report = self.report_generator.generate_execution_report(
                context, metrics, analysis_result, alerts
            )
            
            logging.info(f"生成执行报告: {report.report_id}")
            
        except Exception as e:
            logging.error(f"综合分析失败: {e}")
            logging.error(traceback.format_exc())
    
    def _perform_comprehensive_analysis(self, context: ExecutionContext, 
                                      metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """执行综合分析"""
        # 性能评估
        performance_evaluation = self.performance_evaluator.evaluate_performance(metrics)
        
        # 效果分析
        effect_analysis = self.effect_analyzer.analyze_execution_effect(context, metrics)
        
        # 优化建议
        optimization_suggestions = self.optimization_advisor.generate_optimization_suggestions(
            context, metrics, effect_analysis
        )
        
        # 合并分析结果
        comprehensive_result = {
            "performance_evaluation": performance_evaluation,
            "effect_analysis": effect_analysis,
            "optimization_suggestions": optimization_suggestions,
            "analysis_timestamp": datetime.now()
        }
        
        return comprehensive_result
    
    def get_execution_status(self, execution_id: str) -> Optional[ExecutionContext]:
        """获取执行状态"""
        return self.execution_cache.get(execution_id)
    
    def get_performance_metrics(self, execution_id: str) -> List[PerformanceMetric]:
        """获取性能指标"""
        return self.metrics_cache.get(execution_id, [])
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃预警"""
        return self.alert_manager.get_active_alerts()
    
    def generate_report(self, execution_id: str, format_type: str = "json") -> Optional[str]:
        """生成报告"""
        if execution_id not in self.execution_cache:
            return None
        
        context = self.execution_cache[execution_id]
        metrics = self.metrics_cache.get(execution_id, [])
        
        # 执行分析
        analysis_result = self._perform_comprehensive_analysis(context, metrics)
        
        # 生成报告
        alerts = self.alert_manager.get_active_alerts()
        report = self.report_generator.generate_execution_report(
            context, metrics, analysis_result, alerts
        )
        
        return self.report_generator.export_report(report, format_type)
    
    def get_optimization_suggestions(self, execution_id: str) -> List[Dict[str, Any]]:
        """获取优化建议"""
        if execution_id not in self.execution_cache:
            return []
        
        context = self.execution_cache[execution_id]
        metrics = self.metrics_cache.get(execution_id, [])
        
        # 执行效果分析
        effect_analysis = self.effect_analyzer.analyze_execution_effect(context, metrics)
        
        # 生成优化建议
        suggestions = self.optimization_advisor.generate_optimization_suggestions(
            context, metrics, effect_analysis
        )
        
        return suggestions
    
    def configure_monitoring(self, config_updates: Dict[str, Any]):
        """配置监控参数"""
        self.config.update(config_updates)
        
        # 更新实时监控器配置
        if "monitor_interval" in config_updates:
            interval = config_updates["monitor_interval"]
            self.real_time_monitor.monitor_interval = interval
        
        logging.info("监控配置已更新")
    
    def add_custom_metric_threshold(self, metric_name: str, thresholds: Dict[str, float]):
        """添加自定义指标阈值"""
        self.performance_evaluator.set_thresholds(metric_name, thresholds)
    
    def add_custom_alert_rule(self, rule_name: str, rule_config: Dict[str, Any]):
        """添加自定义预警规则"""
        self.alert_manager.add_alert_rule(rule_name, rule_config)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "is_initialized": self.is_initialized,
            "is_monitoring": self.is_monitoring,
            "active_executions": len(self.real_time_monitor.active_executions),
            "cached_executions": len(self.execution_cache),
            "total_metrics": sum(len(metrics) for metrics in self.metrics_cache.values()),
            "active_alerts": len(self.alert_manager.active_alerts),
            "monitor_interval": self.real_time_monitor.monitor_interval,
            "uptime": datetime.now().isoformat()
        }


# 使用示例和测试代码
def demo_usage():
    """演示用法"""
    # 创建监控器
    monitor = ExecutionMonitor({
        "monitor_interval": 0.5
    })
    
    try:
        # 初始化
        monitor.initialize()
        
        # 创建执行上下文
        execution_context = ExecutionContext(
            execution_id="test_execution_001",
            task_name="数据处理任务",
            start_time=datetime.now(),
            parameters={"timeout": 300}
        )
        
        # 开始监控
        monitor.start_execution_monitoring(execution_context)
        
        # 模拟执行过程
        import random
        for i in range(10):
            # 模拟性能指标
            cpu_usage = random.uniform(30, 90)
            memory_usage = random.uniform(40, 85)
            execution_time = random.uniform(1, 15)
            error_rate = random.uniform(0, 0.1)
            
            # 记录指标
            monitor.record_metric(execution_context.execution_id, PerformanceMetric(
                metric_name="cpu_usage",
                value=cpu_usage,
                unit="%",
                timestamp=datetime.now(),
                metric_type=MetricType.RESOURCE
            ))
            
            monitor.record_metric(execution_context.execution_id, PerformanceMetric(
                metric_name="memory_usage",
                value=memory_usage,
                unit="%",
                timestamp=datetime.now(),
                metric_type=MetricType.RESOURCE
            ))
            
            monitor.record_metric(execution_context.execution_id, PerformanceMetric(
                metric_name="execution_time",
                value=execution_time,
                unit="秒",
                timestamp=datetime.now(),
                metric_type=MetricType.PERFORMANCE
            ))
            
            monitor.record_metric(execution_context.execution_id, PerformanceMetric(
                metric_name="error_rate",
                value=error_rate,
                unit="比例",
                timestamp=datetime.now(),
                metric_type=MetricType.ERROR
            ))
            
            time.sleep(0.5)
        
        # 完成执行
        monitor.update_execution_status(execution_context.execution_id, ExecutionStatus.COMPLETED)
        
        # 等待分析完成
        time.sleep(2)
        
        # 生成报告
        report = monitor.generate_report(execution_context.execution_id)
        if report:
            print("=== 执行报告 ===")
            print(report)
        
        # 获取优化建议
        suggestions = monitor.get_optimization_suggestions(execution_context.execution_id)
        print("\n=== 优化建议 ===")
        for suggestion in suggestions:
            print(f"建议: {suggestion.get('title', '未知')}")
            print(f"描述: {suggestion.get('description', '')}")
            print(f"优先级: {suggestion.get('priority', '未知')}")
            print("---")
        
        # 获取系统状态
        status = monitor.get_system_status()
        print("\n=== 系统状态 ===")
        print(json.dumps(status, indent=2, ensure_ascii=False, default=str))
        
    finally:
        # 关闭监控器
        monitor.shutdown()


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行演示
    demo_usage()