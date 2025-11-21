#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V9模型状态聚合器

该模块实现了一个全面的模型状态聚合器，用于收集、监控、分析和报告模型状态信息。
支持多种模型类型的状态收集、性能指标聚合、健康度评估、资源监控等功能。

主要功能:
- 模型状态收集和聚合
- 性能指标监控和分析
- 模型使用统计
- 健康度评估
- 资源消耗监控
- 版本状态管理
- 部署状态监控
- 状态报告生成
- 告警系统


创建时间: 2025-11-05
版本: V9.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """模型状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPLOYING = "deploying"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    UPDATING = "updating"


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """健康状态枚举"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class ModelInfo:
    """模型信息数据类"""
    model_id: str
    model_name: str
    model_type: str
    version: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    model_id: str
    timestamp: datetime
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    latency: Optional[float] = None
    throughput: Optional[float] = None
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    error_rate: Optional[float] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageStatistics:
    """使用统计数据类"""
    model_id: str
    timestamp: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    peak_concurrent_users: int = 0
    daily_active_users: int = 0
    user_feedback_score: Optional[float] = None


@dataclass
class ResourceConsumption:
    """资源消耗数据类"""
    model_id: str
    timestamp: datetime
    cpu_cores: float = 0.0
    memory_gb: float = 0.0
    gpu_memory_gb: float = 0.0
    storage_gb: float = 0.0
    network_mbps: float = 0.0
    power_consumption_watts: Optional[float] = None


@dataclass
class VersionInfo:
    """版本信息数据类"""
    model_id: str
    version: str
    release_date: datetime
    status: ModelStatus
    deployment_status: str
    changelog: str = ""
    rollback_available: bool = False


@dataclass
class HealthAssessment:
    """健康评估数据类"""
    model_id: str
    timestamp: datetime
    overall_health: HealthStatus
    performance_score: float
    reliability_score: float
    efficiency_score: float
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """告警数据类"""
    alert_id: str
    model_id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class ModelStateAggregator:
    """
    V9模型状态聚合器
    
    该类提供全面的模型状态管理功能，包括状态收集、性能监控、
    健康评估、资源监控、版本管理和告警系统。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化模型状态聚合器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        self.models: Dict[str, ModelInfo] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.usage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.resource_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.version_history: Dict[str, List[VersionInfo]] = defaultdict(list)
        self.health_assessments: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # 监控配置
        self.monitoring_interval = self.config.get('monitoring_interval', 60)  # 秒
        self.health_check_interval = self.config.get('health_check_interval', 300)  # 秒
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        
        # 线程管理
        self._monitoring_active = False
        self._monitoring_thread = None
        self._lock = threading.RLock()
        
        logger.info("模型状态聚合器初始化完成")
    
    # ==================== 模型状态收集 ====================
    
    def register_model(self, model_info: ModelInfo) -> bool:
        """
        注册新模型
        
        Args:
            model_info: 模型信息
            
        Returns:
            bool: 注册是否成功
        """
        try:
            with self._lock:
                if model_info.model_id in self.models:
                    logger.warning(f"模型 {model_info.model_id} 已存在，将更新信息")
                
                self.models[model_info.model_id] = model_info
                logger.info(f"模型 {model_info.model_id} 注册成功")
                return True
                
        except Exception as e:
            logger.error(f"注册模型失败: {e}")
            return False
    
    def unregister_model(self, model_id: str) -> bool:
        """
        注销模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            bool: 注销是否成功
        """
        try:
            with self._lock:
                if model_id not in self.models:
                    logger.warning(f"模型 {model_id} 不存在")
                    return False
                
                del self.models[model_id]
                # 清理相关历史数据
                self.performance_history.pop(model_id, None)
                self.usage_history.pop(model_id, None)
                self.resource_history.pop(model_id, None)
                self.version_history.pop(model_id, None)
                self.health_assessments.pop(model_id, None)
                
                logger.info(f"模型 {model_id} 注销成功")
                return True
                
        except Exception as e:
            logger.error(f"注销模型失败: {e}")
            return False
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        获取模型信息
        
        Args:
            model_id: 模型ID
            
        Returns:
            ModelInfo: 模型信息，如果不存在返回None
        """
        return self.models.get(model_id)
    
    def list_models(self, status: Optional[ModelStatus] = None) -> List[ModelInfo]:
        """
        列出模型列表
        
        Args:
            status: 可选的状态过滤条件
            
        Returns:
            List[ModelInfo]: 模型信息列表
        """
        with self._lock:
            models = list(self.models.values())
            if status:
                models = [m for m in models if m.status == status]
            return models
    
    # ==================== 性能指标聚合 ====================
    
    def record_performance_metrics(self, metrics: PerformanceMetrics) -> bool:
        """
        记录性能指标
        
        Args:
            metrics: 性能指标数据
            
        Returns:
            bool: 记录是否成功
        """
        try:
            with self._lock:
                if metrics.model_id not in self.models:
                    logger.warning(f"模型 {metrics.model_id} 不存在")
                    return False
                
                self.performance_history[metrics.model_id].append(metrics)
                logger.debug(f"性能指标记录成功: {metrics.model_id}")
                return True
                
        except Exception as e:
            logger.error(f"记录性能指标失败: {e}")
            return False
    
    def get_performance_summary(self, model_id: str, 
                              time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        获取性能摘要
        
        Args:
            model_id: 模型ID
            time_range: 时间范围
            
        Returns:
            Dict[str, Any]: 性能摘要
        """
        try:
            with self._lock:
                if model_id not in self.performance_history:
                    return {}
                
                history = list(self.performance_history[model_id])
                
                # 时间过滤
                if time_range:
                    cutoff_time = datetime.now() - time_range
                    history = [h for h in history if h.timestamp >= cutoff_time]
                
                if not history:
                    return {}
                
                # 计算统计指标
                summary = {
                    'model_id': model_id,
                    'data_points': len(history),
                    'time_range': {
                        'start': min(h.timestamp for h in history),
                        'end': max(h.timestamp for h in history)
                    },
                    'metrics': {}
                }
                
                # 数值指标统计
                numeric_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 
                                 'latency', 'throughput', 'cpu_usage', 'memory_usage', 
                                 'gpu_usage', 'error_rate']
                
                for metric in numeric_metrics:
                    values = [getattr(h, metric) for h in history 
                             if getattr(h, metric) is not None]
                    if values:
                        summary['metrics'][metric] = {
                            'mean': statistics.mean(values),
                            'median': statistics.median(values),
                            'min': min(values),
                            'max': max(values),
                            'std': statistics.stdev(values) if len(values) > 1 else 0.0
                        }
                
                return summary
                
        except Exception as e:
            logger.error(f"获取性能摘要失败: {e}")
            return {}
    
    def aggregate_performance_across_models(self, 
                                          time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        聚合所有模型的性能指标
        
        Args:
            time_range: 时间范围
            
        Returns:
            Dict[str, Any]: 聚合性能指标
        """
        try:
            with self._lock:
                aggregation = {
                    'timestamp': datetime.now(),
                    'total_models': len(self.models),
                    'models_with_data': 0,
                    'overall_metrics': {},
                    'model_summaries': []
                }
                
                all_metrics = defaultdict(list)
                
                for model_id in self.models.keys():
                    summary = self.get_performance_summary(model_id, time_range)
                    if summary:
                        aggregation['models_with_data'] += 1
                        aggregation['model_summaries'].append(summary)
                        
                        # 收集所有模型的指标用于整体聚合
                        for metric_name, metric_data in summary.get('metrics', {}).items():
                            all_metrics[metric_name].append(metric_data['mean'])
                
                # 计算整体指标
                for metric_name, values in all_metrics.items():
                    aggregation['overall_metrics'][metric_name] = {
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'min': min(values),
                        'max': max(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0
                    }
                
                return aggregation
                
        except Exception as e:
            logger.error(f"聚合性能指标失败: {e}")
            return {}
    
    # ==================== 模型使用统计 ====================
    
    def record_usage_statistics(self, usage: UsageStatistics) -> bool:
        """
        记录使用统计
        
        Args:
            usage: 使用统计数据
            
        Returns:
            bool: 记录是否成功
        """
        try:
            with self._lock:
                if usage.model_id not in self.models:
                    logger.warning(f"模型 {usage.model_id} 不存在")
                    return False
                
                self.usage_history[usage.model_id].append(usage)
                logger.debug(f"使用统计记录成功: {usage.model_id}")
                return True
                
        except Exception as e:
            logger.error(f"记录使用统计失败: {e}")
            return False
    
    def get_usage_summary(self, model_id: str, 
                         time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        获取使用统计摘要
        
        Args:
            model_id: 模型ID
            time_range: 时间范围
            
        Returns:
            Dict[str, Any]: 使用统计摘要
        """
        try:
            with self._lock:
                if model_id not in self.usage_history:
                    return {}
                
                history = list(self.usage_history[model_id])
                
                # 时间过滤
                if time_range:
                    cutoff_time = datetime.now() - time_range
                    history = [h for h in history if h.timestamp >= cutoff_time]
                
                if not history:
                    return {}
                
                # 计算统计指标
                total_requests = sum(h.total_requests for h in history)
                successful_requests = sum(h.successful_requests for h in history)
                failed_requests = sum(h.failed_requests for h in history)
                
                summary = {
                    'model_id': model_id,
                    'period': {
                        'start': min(h.timestamp for h in history),
                        'end': max(h.timestamp for h in history),
                        'days': (max(h.timestamp for h in history) - 
                               min(h.timestamp for h in history)).days + 1
                    },
                    'total_requests': total_requests,
                    'successful_requests': successful_requests,
                    'failed_requests': failed_requests,
                    'success_rate': (successful_requests / total_requests * 100 
                                   if total_requests > 0 else 0),
                    'average_response_time': statistics.mean([h.average_response_time 
                                                             for h in history]),
                    'peak_concurrent_users': max(h.peak_concurrent_users for h in history),
                    'daily_active_users': statistics.mean([h.daily_active_users 
                                                         for h in history])
                }
                
                return summary
                
        except Exception as e:
            logger.error(f"获取使用统计摘要失败: {e}")
            return {}
    
    # ==================== 模型健康度评估 ====================
    
    def assess_model_health(self, model_id: str) -> Optional[HealthAssessment]:
        """
        评估模型健康度
        
        Args:
            model_id: 模型ID
            
        Returns:
            HealthAssessment: 健康评估结果
        """
        try:
            with self._lock:
                if model_id not in self.models:
                    logger.warning(f"模型 {model_id} 不存在")
                    return None
                
                # 获取最近的性能数据
                recent_performance = list(self.performance_history[model_id])[-10:]
                recent_usage = list(self.usage_history[model_id])[-10:]
                
                # 计算各项得分
                performance_score = self._calculate_performance_score(recent_performance)
                reliability_score = self._calculate_reliability_score(recent_performance, recent_usage)
                efficiency_score = self._calculate_efficiency_score(recent_performance, recent_usage)
                
                # 综合得分
                overall_score = (performance_score * 0.4 + 
                               reliability_score * 0.4 + 
                               efficiency_score * 0.2)
                
                # 确定健康状态
                if overall_score >= 90:
                    health_status = HealthStatus.EXCELLENT
                elif overall_score >= 80:
                    health_status = HealthStatus.GOOD
                elif overall_score >= 70:
                    health_status = HealthStatus.FAIR
                elif overall_score >= 50:
                    health_status = HealthStatus.POOR
                else:
                    health_status = HealthStatus.CRITICAL
                
                # 生成问题和建议
                issues, recommendations = self._generate_health_recommendations(
                    model_id, overall_score, recent_performance, recent_usage)
                
                assessment = HealthAssessment(
                    model_id=model_id,
                    timestamp=datetime.now(),
                    overall_health=health_status,
                    performance_score=performance_score,
                    reliability_score=reliability_score,
                    efficiency_score=efficiency_score,
                    issues=issues,
                    recommendations=recommendations
                )
                
                self.health_assessments[model_id].append(assessment)
                logger.info(f"模型 {model_id} 健康度评估完成: {health_status.value}")
                return assessment
                
        except Exception as e:
            logger.error(f"评估模型健康度失败: {e}")
            return None
    
    def _calculate_performance_score(self, performance_data: List[PerformanceMetrics]) -> float:
        """计算性能得分"""
        if not performance_data:
            return 0.0
        
        scores = []
        
        # 准确率得分
        accuracies = [p.accuracy for p in performance_data if p.accuracy is not None]
        if accuracies:
            scores.append(statistics.mean(accuracies) * 100)
        
        # 延迟得分 (延迟越低得分越高)
        latencies = [p.latency for p in performance_data if p.latency is not None]
        if latencies:
            avg_latency = statistics.mean(latencies)
            # 假设100ms为满分，1000ms为0分
            latency_score = max(0, 100 - (avg_latency - 100) / 9)
            scores.append(latency_score)
        
        # 吞吐量得分
        throughputs = [p.throughput for p in performance_data if p.throughput is not None]
        if throughputs:
            scores.append(min(100, statistics.mean(throughputs)))
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_reliability_score(self, performance_data: List[PerformanceMetrics], 
                                   usage_data: List[UsageStatistics]) -> float:
        """计算可靠性得分"""
        scores = []
        
        # 错误率得分
        error_rates = [p.error_rate for p in performance_data if p.error_rate is not None]
        if error_rates:
            avg_error_rate = statistics.mean(error_rates)
            scores.append(max(0, 100 - avg_error_rate * 100))
        
        # 成功率得分
        if usage_data:
            total_requests = sum(u.total_requests for u in usage_data)
            successful_requests = sum(u.successful_requests for u in usage_data)
            if total_requests > 0:
                success_rate = (successful_requests / total_requests) * 100
                scores.append(success_rate)
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_efficiency_score(self, performance_data: List[PerformanceMetrics], 
                                  usage_data: List[UsageStatistics]) -> float:
        """计算效率得分"""
        scores = []
        
        # CPU使用率得分 (适度使用得分最高)
        cpu_usages = [p.cpu_usage for p in performance_data if p.cpu_usage is not None]
        if cpu_usages:
            avg_cpu = statistics.mean(cpu_usages)
            # 60-80%为最佳范围
            if 60 <= avg_cpu <= 80:
                cpu_score = 100
            else:
                cpu_score = max(0, 100 - abs(avg_cpu - 70) * 2)
            scores.append(cpu_score)
        
        # 内存使用率得分
        memory_usages = [p.memory_usage for p in performance_data if p.memory_usage is not None]
        if memory_usages:
            avg_memory = statistics.mean(memory_usages)
            # 70-85%为最佳范围
            if 70 <= avg_memory <= 85:
                memory_score = 100
            else:
                memory_score = max(0, 100 - abs(avg_memory - 77.5) * 2)
            scores.append(memory_score)
        
        return statistics.mean(scores) if scores else 0.0
    
    def _generate_health_recommendations(self, model_id: str, overall_score: float,
                                       performance_data: List[PerformanceMetrics],
                                       usage_data: List[UsageStatistics]) -> Tuple[List[str], List[str]]:
        """生成健康建议"""
        issues = []
        recommendations = []
        
        if overall_score < 70:
            issues.append("模型整体性能需要改进")
        
        if performance_data:
            recent_perf = performance_data[-1]
            
            if recent_perf.accuracy and recent_perf.accuracy < 0.8:
                issues.append("模型准确率偏低")
                recommendations.append("考虑重新训练模型或调整超参数")
            
            if recent_perf.latency and recent_perf.latency > 1000:
                issues.append("模型响应时间过长")
                recommendations.append("优化模型推理速度或增加计算资源")
            
            if recent_perf.error_rate and recent_perf.error_rate > 0.05:
                issues.append("模型错误率较高")
                recommendations.append("检查数据质量和模型稳定性")
        
        if usage_data:
            recent_usage = usage_data[-1]
            success_rate = (recent_usage.successful_requests / recent_usage.total_requests * 100 
                          if recent_usage.total_requests > 0 else 0)
            if success_rate < 95:
                issues.append("模型成功率偏低")
                recommendations.append("检查模型部署状态和错误处理机制")
        
        if not recommendations:
            recommendations.append("模型运行状态良好，继续保持")
        
        return issues, recommendations
    
    # ==================== 模型资源消耗监控 ====================
    
    def record_resource_consumption(self, consumption: ResourceConsumption) -> bool:
        """
        记录资源消耗
        
        Args:
            consumption: 资源消耗数据
            
        Returns:
            bool: 记录是否成功
        """
        try:
            with self._lock:
                if consumption.model_id not in self.models:
                    logger.warning(f"模型 {consumption.model_id} 不存在")
                    return False
                
                self.resource_history[consumption.model_id].append(consumption)
                logger.debug(f"资源消耗记录成功: {consumption.model_id}")
                return True
                
        except Exception as e:
            logger.error(f"记录资源消耗失败: {e}")
            return False
    
    def get_resource_summary(self, model_id: str, 
                           time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        获取资源消耗摘要
        
        Args:
            model_id: 模型ID
            time_range: 时间范围
            
        Returns:
            Dict[str, Any]: 资源消耗摘要
        """
        try:
            with self._lock:
                if model_id not in self.resource_history:
                    return {}
                
                history = list(self.resource_history[model_id])
                
                # 时间过滤
                if time_range:
                    cutoff_time = datetime.now() - time_range
                    history = [h for h in history if h.timestamp >= cutoff_time]
                
                if not history:
                    return {}
                
                # 计算统计指标
                summary = {
                    'model_id': model_id,
                    'period': {
                        'start': min(h.timestamp for h in history),
                        'end': max(h.timestamp for h in history)
                    },
                    'cpu_cores': {
                        'average': statistics.mean([h.cpu_cores for h in history]),
                        'max': max([h.cpu_cores for h in history]),
                        'min': min([h.cpu_cores for h in history])
                    },
                    'memory_gb': {
                        'average': statistics.mean([h.memory_gb for h in history]),
                        'max': max([h.memory_gb for h in history]),
                        'min': min([h.memory_gb for h in history])
                    },
                    'gpu_memory_gb': {
                        'average': statistics.mean([h.gpu_memory_gb for h in history]),
                        'max': max([h.gpu_memory_gb for h in history]),
                        'min': min([h.gpu_memory_gb for h in history])
                    },
                    'storage_gb': {
                        'average': statistics.mean([h.storage_gb for h in history]),
                        'max': max([h.storage_gb for h in history]),
                        'min': min([h.storage_gb for h in history])
                    }
                }
                
                return summary
                
        except Exception as e:
            logger.error(f"获取资源消耗摘要失败: {e}")
            return {}
    
    # ==================== 模型版本状态管理 ====================
    
    def register_model_version(self, version_info: VersionInfo) -> bool:
        """
        注册模型版本
        
        Args:
            version_info: 版本信息
            
        Returns:
            bool: 注册是否成功
        """
        try:
            with self._lock:
                if version_info.model_id not in self.models:
                    logger.warning(f"模型 {version_info.model_id} 不存在")
                    return False
                
                self.version_history[version_info.model_id].append(version_info)
                # 按版本号排序
                self.version_history[version_info.model_id].sort(
                    key=lambda v: v.version, reverse=True)
                
                logger.info(f"模型版本注册成功: {version_info.model_id} v{version_info.version}")
                return True
                
        except Exception as e:
            logger.error(f"注册模型版本失败: {e}")
            return False
    
    def get_version_history(self, model_id: str) -> List[VersionInfo]:
        """
        获取版本历史
        
        Args:
            model_id: 模型ID
            
        Returns:
            List[VersionInfo]: 版本历史列表
        """
        return self.version_history.get(model_id, [])
    
    def get_current_version(self, model_id: str) -> Optional[VersionInfo]:
        """
        获取当前版本
        
        Args:
            model_id: 模型ID
            
        Returns:
            VersionInfo: 当前版本信息
        """
        versions = self.get_version_history(model_id)
        return versions[0] if versions else None
    
    # ==================== 模型部署状态监控 ====================
    
    def update_deployment_status(self, model_id: str, status: ModelStatus, 
                               additional_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        更新部署状态
        
        Args:
            model_id: 模型ID
            status: 部署状态
            additional_info: 附加信息
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with self._lock:
                if model_id not in self.models:
                    logger.warning(f"模型 {model_id} 不存在")
                    return False
                
                # 更新模型状态
                self.models[model_id].status = status
                self.models[model_id].updated_at = datetime.now()
                
                # 记录状态变更
                logger.info(f"模型 {model_id} 状态更新为: {status.value}")
                
                # 如果是错误状态，生成告警
                if status == ModelStatus.ERROR:
                    self._create_alert(
                        model_id=model_id,
                        level=AlertLevel.ERROR,
                        message=f"模型 {model_id} 进入错误状态"
                    )
                
                return True
                
        except Exception as e:
            logger.error(f"更新部署状态失败: {e}")
            return False
    
    # ==================== 模型状态报告生成 ====================
    
    def generate_comprehensive_report(self, model_id: Optional[str] = None,
                                    time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        生成综合状态报告
        
        Args:
            model_id: 可选的模型ID，如果为None则生成所有模型的报告
            time_range: 时间范围
            
        Returns:
            Dict[str, Any]: 综合报告
        """
        try:
            report = {
                'report_id': f"report_{int(time.time())}",
                'generated_at': datetime.now().isoformat(),
                'time_range': time_range.__str__() if time_range else "all_time",
                'scope': 'single_model' if model_id else 'all_models'
            }
            
            if model_id:
                # 单个模型报告
                report.update(self._generate_single_model_report(model_id, time_range))
            else:
                # 所有模型报告
                report.update(self._generate_all_models_report(time_range))
            
            logger.info(f"综合报告生成完成: {report['report_id']}")
            return report
            
        except Exception as e:
            logger.error(f"生成综合报告失败: {e}")
            return {}
    
    def _generate_single_model_report(self, model_id: str, 
                                    time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """生成单个模型报告"""
        with self._lock:
            if model_id not in self.models:
                return {'error': f'模型 {model_id} 不存在'}
            
            model_info = self.models[model_id]
            
            # 基本信息
            report = {
                'model_info': asdict(model_info),
                'performance_summary': self.get_performance_summary(model_id, time_range),
                'usage_summary': self.get_usage_summary(model_id, time_range),
                'resource_summary': self.get_resource_summary(model_id, time_range),
                'version_history': [asdict(v) for v in self.get_version_history(model_id)],
                'current_version': asdict(self.get_current_version(model_id)) if self.get_current_version(model_id) else None
            }
            
            # 健康评估
            health_assessment = self.assess_model_health(model_id)
            if health_assessment:
                report['health_assessment'] = asdict(health_assessment)
            
            # 最近告警
            recent_alerts = [asdict(alert) for alert in self.alerts 
                           if alert.model_id == model_id and not alert.resolved][-5:]
            report['recent_alerts'] = recent_alerts
            
            return report
    
    def _generate_all_models_report(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """生成所有模型报告"""
        with self._lock:
            report = {
                'total_models': len(self.models),
                'models_by_status': {},
                'performance_aggregation': self.aggregate_performance_across_models(time_range),
                'model_reports': []
            }
            
            # 按状态统计模型数量
            for status in ModelStatus:
                status_models = [m for m in self.models.values() if m.status == status]
                report['models_by_status'][status.value] = len(status_models)
            
            # 生成每个模型的简要报告
            for model_id in self.models.keys():
                model_report = self._generate_single_model_report(model_id, time_range)
                model_report['model_id'] = model_id
                report['model_reports'].append(model_report)
            
            # 系统整体健康状况
            health_scores = []
            for model_id in self.models.keys():
                health = self.assess_model_health(model_id)
                if health:
                    health_scores.append(
                        (health.performance_score + health.reliability_score + health.efficiency_score) / 3
                    )
            
            if health_scores:
                report['system_health'] = {
                    'average_health_score': statistics.mean(health_scores),
                    'min_health_score': min(health_scores),
                    'max_health_score': max(health_scores)
                }
            
            return report
    
    def export_report_to_json(self, report: Dict[str, Any], 
                            file_path: str) -> bool:
        """
        将报告导出为JSON文件
        
        Args:
            report: 报告数据
            file_path: 输出文件路径
            
        Returns:
            bool: 导出是否成功
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"报告已导出到: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出报告失败: {e}")
            return False
    
    # ==================== 模型状态告警 ====================
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        添加告警回调函数
        
        Args:
            callback: 告警回调函数
        """
        self.alert_callbacks.append(callback)
        logger.info("告警回调函数添加成功")
    
    def _create_alert(self, model_id: str, level: AlertLevel, message: str) -> Alert:
        """
        创建告警
        
        Args:
            model_id: 模型ID
            level: 告警级别
            message: 告警消息
            
        Returns:
            Alert: 告警对象
        """
        alert = Alert(
            alert_id=f"alert_{int(time.time())}_{model_id}",
            model_id=model_id,
            level=level,
            message=message,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        
        # 触发回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
        
        logger.warning(f"告警创建: {alert.alert_id} - {message}")
        return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        解决告警
        
        Args:
            alert_id: 告警ID
            
        Returns:
            bool: 解决是否成功
        """
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    logger.info(f"告警已解决: {alert_id}")
                    return True
            
            logger.warning(f"告警未找到或已解决: {alert_id}")
            return False
            
        except Exception as e:
            logger.error(f"解决告警失败: {e}")
            return False
    
    def get_active_alerts(self, model_id: Optional[str] = None) -> List[Alert]:
        """
        获取活跃告警
        
        Args:
            model_id: 可选的模型ID过滤条件
            
        Returns:
            List[Alert]: 活跃告警列表
        """
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        if model_id:
            active_alerts = [alert for alert in active_alerts if alert.model_id == model_id]
        
        return active_alerts
    
    # ==================== 监控管理 ====================
    
    def start_monitoring(self) -> None:
        """启动监控服务"""
        if self._monitoring_active:
            logger.warning("监控服务已在运行")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("监控服务已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控服务"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("监控服务已停止")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self._monitoring_active:
            try:
                # 执行健康检查
                self._perform_health_checks()
                
                # 检查告警条件
                self._check_alert_conditions()
                
                # 清理过期数据
                self._cleanup_expired_data()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(5)
    
    def _perform_health_checks(self) -> None:
        """执行健康检查"""
        for model_id in list(self.models.keys()):
            try:
                self.assess_model_health(model_id)
            except Exception as e:
                logger.error(f"模型 {model_id} 健康检查失败: {e}")
    
    def _check_alert_conditions(self) -> None:
        """检查告警条件"""
        for model_id in self.models.keys():
            try:
                # 检查性能告警
                recent_perf = list(self.performance_history[model_id])[-1:] if self.performance_history[model_id] else []
                if recent_perf:
                    perf = recent_perf[0]
                    
                    # CPU使用率告警
                    if perf.cpu_usage and perf.cpu_usage > 90:
                        self._create_alert(
                            model_id=model_id,
                            level=AlertLevel.WARNING,
                            message=f"模型 {model_id} CPU使用率过高: {perf.cpu_usage}%"
                        )
                    
                    # 内存使用率告警
                    if perf.memory_usage and perf.memory_usage > 90:
                        self._create_alert(
                            model_id=model_id,
                            level=AlertLevel.WARNING,
                            message=f"模型 {model_id} 内存使用率过高: {perf.memory_usage}%"
                        )
                    
                    # 错误率告警
                    if perf.error_rate and perf.error_rate > 0.1:
                        self._create_alert(
                            model_id=model_id,
                            level=AlertLevel.ERROR,
                            message=f"模型 {model_id} 错误率过高: {perf.error_rate*100}%"
                        )
                
            except Exception as e:
                logger.error(f"检查模型 {model_id} 告警条件失败: {e}")
    
    def _cleanup_expired_data(self) -> None:
        """清理过期数据"""
        # 清理30天前的告警
        cutoff_time = datetime.now() - timedelta(days=30)
        self.alerts = [alert for alert in self.alerts 
                      if alert.timestamp >= cutoff_time or not alert.resolved]
        
        logger.debug("过期数据清理完成")


# ==================== 测试用例 ====================

class ModelStateAggregatorTest:
    """模型状态聚合器测试类"""
    
    @staticmethod
    def run_all_tests():
        """运行所有测试"""
        print("开始运行模型状态聚合器测试...")
        
        test_instance = ModelStateAggregatorTest()
        
        try:
            test_instance.test_basic_operations()
            test_instance.test_performance_metrics()
            test_instance.test_health_assessment()
            test_instance.test_alert_system()
            test_instance.test_report_generation()
            test_instance.test_monitoring()
            
            print("✅ 所有测试通过!")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            raise
    
    def test_basic_operations(self):
        """测试基本操作"""
        print("测试基本操作...")
        
        # 创建聚合器
        aggregator = ModelStateAggregator()
        
        # 注册模型
        model_info = ModelInfo(
            model_id="test_model_1",
            model_name="测试模型1",
            model_type="classification",
            version="1.0.0",
            status=ModelStatus.ACTIVE,
            created_at=datetime.now(),
            description="用于测试的分类模型"
        )
        
        assert aggregator.register_model(model_info), "模型注册失败"
        assert aggregator.get_model_info("test_model_1") is not None, "获取模型信息失败"
        
        # 列出模型
        models = aggregator.list_models()
        assert len(models) == 1, "模型列表不正确"
        
        print("✅ 基本操作测试通过")
    
    def test_performance_metrics(self):
        """测试性能指标"""
        print("测试性能指标...")
        
        aggregator = ModelStateAggregator()
        
        # 注册模型
        model_info = ModelInfo(
            model_id="perf_model",
            model_name="性能测试模型",
            model_type="regression",
            version="1.0.0",
            status=ModelStatus.ACTIVE,
            created_at=datetime.now()
        )
        aggregator.register_model(model_info)
        
        # 记录性能指标
        metrics = PerformanceMetrics(
            model_id="perf_model",
            timestamp=datetime.now(),
            accuracy=0.95,
            latency=120.5,
            throughput=100.0,
            cpu_usage=65.0,
            memory_usage=70.0,
            error_rate=0.02
        )
        
        assert aggregator.record_performance_metrics(metrics), "记录性能指标失败"
        
        # 获取性能摘要
        summary = aggregator.get_performance_summary("perf_model")
        assert summary is not None, "获取性能摘要失败"
        assert summary['metrics']['accuracy']['mean'] == 0.95, "性能指标不正确"
        
        print("✅ 性能指标测试通过")
    
    def test_health_assessment(self):
        """测试健康评估"""
        print("测试健康评估...")
        
        aggregator = ModelStateAggregator()
        
        # 注册模型
        model_info = ModelInfo(
            model_id="health_model",
            model_name="健康测试模型",
            model_type="classification",
            version="1.0.0",
            status=ModelStatus.ACTIVE,
            created_at=datetime.now()
        )
        aggregator.register_model(model_info)
        
        # 记录性能指标
        for i in range(5):
            metrics = PerformanceMetrics(
                model_id="health_model",
                timestamp=datetime.now() - timedelta(minutes=i),
                accuracy=0.90 + i * 0.01,
                latency=100 + i * 10,
                cpu_usage=60 + i * 5,
                memory_usage=65 + i * 3,
                error_rate=0.01 + i * 0.005
            )
            aggregator.record_performance_metrics(metrics)
        
        # 记录使用统计
        usage = UsageStatistics(
            model_id="health_model",
            timestamp=datetime.now(),
            total_requests=1000,
            successful_requests=980,
            failed_requests=20,
            average_response_time=105.0,
            peak_concurrent_users=50
        )
        aggregator.record_usage_statistics(usage)
        
        # 评估健康度
        health = aggregator.assess_model_health("health_model")
        assert health is not None, "健康评估失败"
        assert health.overall_health in [HealthStatus.EXCELLENT, HealthStatus.GOOD, 
                                       HealthStatus.FAIR, HealthStatus.POOR, HealthStatus.CRITICAL], \
               "健康状态不正确"
        
        print("✅ 健康评估测试通过")
    
    def test_alert_system(self):
        """测试告警系统"""
        print("测试告警系统...")
        
        aggregator = ModelStateAggregator()
        
        # 注册模型
        model_info = ModelInfo(
            model_id="alert_model",
            model_name="告警测试模型",
            model_type="classification",
            version="1.0.0",
            status=ModelStatus.ACTIVE,
            created_at=datetime.now()
        )
        aggregator.register_model(model_info)
        
        # 添加告警回调
        alert_received = []
        
        def alert_callback(alert):
            alert_received.append(alert)
        
        aggregator.add_alert_callback(alert_callback)
        
        # 创建告警
        alert = aggregator._create_alert(
            model_id="alert_model",
            level=AlertLevel.WARNING,
            message="测试告警消息"
        )
        
        assert len(alert_received) == 1, "告警回调未触发"
        assert alert_received[0].level == AlertLevel.WARNING, "告警级别不正确"
        
        # 获取活跃告警
        active_alerts = aggregator.get_active_alerts()
        assert len(active_alerts) == 1, "活跃告警数量不正确"
        
        # 解决告警
        assert aggregator.resolve_alert(alert.alert_id), "解决告警失败"
        
        # 检查告警是否已解决
        active_alerts_after = aggregator.get_active_alerts()
        assert len(active_alerts_after) == 0, "告警未正确解决"
        
        print("✅ 告警系统测试通过")
    
    def test_report_generation(self):
        """测试报告生成"""
        print("测试报告生成...")
        
        aggregator = ModelStateAggregator()
        
        # 注册模型
        model_info = ModelInfo(
            model_id="report_model",
            model_name="报告测试模型",
            model_type="classification",
            version="1.0.0",
            status=ModelStatus.ACTIVE,
            created_at=datetime.now()
        )
        aggregator.register_model(model_info)
        
        # 记录一些数据
        metrics = PerformanceMetrics(
            model_id="report_model",
            timestamp=datetime.now(),
            accuracy=0.92,
            latency=110.0
        )
        aggregator.record_performance_metrics(metrics)
        
        # 生成单个模型报告
        single_report = aggregator.generate_comprehensive_report("report_model")
        assert single_report is not None, "生成单个模型报告失败"
        assert 'model_info' in single_report, "报告缺少模型信息"
        
        # 生成所有模型报告
        all_report = aggregator.generate_comprehensive_report()
        assert all_report is not None, "生成所有模型报告失败"
        assert 'total_models' in all_report, "报告缺少模型总数"
        
        print("✅ 报告生成测试通过")
    
    def test_monitoring(self):
        """测试监控功能"""
        print("测试监控功能...")
        
        aggregator = ModelStateAggregator()
        
        # 注册模型
        model_info = ModelInfo(
            model_id="monitor_model",
            model_name="监控测试模型",
            model_type="classification",
            version="1.0.0",
            status=ModelStatus.ACTIVE,
            created_at=datetime.now()
        )
        aggregator.register_model(model_info)
        
        # 启动监控
        aggregator.start_monitoring()
        assert aggregator._monitoring_active, "监控服务未启动"
        
        # 等待一段时间
        time.sleep(2)
        
        # 停止监控
        aggregator.stop_monitoring()
        assert not aggregator._monitoring_active, "监控服务未停止"
        
        print("✅ 监控功能测试通过")


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    print("模型状态聚合器使用示例")
    
    # 创建聚合器
    aggregator = ModelStateAggregator({
        'monitoring_interval': 30,
        'health_check_interval': 300
    })
    
    # 注册模型
    model = ModelInfo(
        model_id="example_model",
        model_name="示例模型",
        model_type="classification",
        version="2.1.0",
        status=ModelStatus.ACTIVE,
        created_at=datetime.now(),
        description="用于文本分类的深度学习模型"
    )
    aggregator.register_model(model)
    
    # 记录性能指标
    metrics = PerformanceMetrics(
        model_id="example_model",
        timestamp=datetime.now(),
        accuracy=0.94,
        precision=0.92,
        recall=0.95,
        f1_score=0.935,
        latency=85.2,
        throughput=150.0,
        cpu_usage=68.5,
        memory_usage=72.3,
        gpu_usage=45.0,
        error_rate=0.015
    )
    aggregator.record_performance_metrics(metrics)
    
    # 记录使用统计
    usage = UsageStatistics(
        model_id="example_model",
        timestamp=datetime.now(),
        total_requests=5000,
        successful_requests=4925,
        failed_requests=75,
        average_response_time=87.3,
        peak_concurrent_users=120,
        daily_active_users=850
    )
    aggregator.record_usage_statistics(usage)
    
    # 记录资源消耗
    resource = ResourceConsumption(
        model_id="example_model",
        timestamp=datetime.now(),
        cpu_cores=4.5,
        memory_gb=8.2,
        gpu_memory_gb=6.0,
        storage_gb=15.5,
        network_mbps=25.0
    )
    aggregator.record_resource_consumption(resource)
    
    # 注册版本信息
    version = VersionInfo(
        model_id="example_model",
        version="2.1.0",
        release_date=datetime.now() - timedelta(days=7),
        status=ModelStatus.ACTIVE,
        deployment_status="production",
        changelog="性能优化和bug修复",
        rollback_available=True
    )
    aggregator.register_model_version(version)
    
    # 评估健康度
    health = aggregator.assess_model_health("example_model")
    if health:
        print(f"模型健康状态: {health.overall_health.value}")
        print(f"性能得分: {health.performance_score:.2f}")
        print(f"可靠性得分: {health.reliability_score:.2f}")
        print(f"效率得分: {health.efficiency_score:.2f}")
    
    # 生成综合报告
    report = aggregator.generate_comprehensive_report("example_model")
    print(f"报告生成完成: {report['report_id']}")
    
    # 添加告警回调
    def handle_alert(alert):
        print(f"告警通知: {alert.level.value} - {alert.message}")
    
    aggregator.add_alert_callback(handle_alert)
    
    # 启动监控
    aggregator.start_monitoring()
    print("监控服务已启动")
    
    # 等待一段时间后停止
    time.sleep(5)
    aggregator.stop_monitoring()
    print("监控服务已停止")


if __name__ == "__main__":
    # 运行测试
    ModelStateAggregatorTest.run_all_tests()
    
    # 运行示例
    print("\n" + "="*50)
    example_usage()