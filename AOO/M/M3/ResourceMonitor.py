#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M3资源监控器 - ResourceMonitor.py

功能描述：
- CPU使用率监控
- 内存使用监控
- 磁盘空间监控
- 网络带宽监控
- GPU使用监控
- 资源限制管理
- 资源预警机制
- 资源成本分析
- 资源优化建议

作者: M3系统
创建时间: 2025-11-05
版本: 1.0.0
"""

import time
import logging
import threading
import json
import psutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import warnings

# 尝试导入GPU监控库
try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    warnings.warn("GPUtil未安装，GPU监控功能将不可用。请运行: pip install GPUtil")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """预警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ResourceType(Enum):
    """资源类型枚举"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"


@dataclass
class ResourceMetrics:
    """资源指标数据类"""
    timestamp: datetime
    resource_type: ResourceType
    usage_percent: float
    total: float
    used: float
    available: float
    unit: str
    additional_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_data is None:
            self.additional_data = {}


@dataclass
class AlertRule:
    """预警规则数据类"""
    resource_type: ResourceType
    threshold: float
    level: AlertLevel
    duration_seconds: int = 0  # 持续时间，0表示立即触发
    enabled: bool = True
    callback: Optional[Callable] = None


@dataclass
class CostAnalysis:
    """成本分析数据类"""
    resource_type: ResourceType
    hourly_cost: float
    daily_cost: float
    monthly_cost: float
    efficiency_score: float  # 效率评分 0-100
    optimization_potential: float  # 优化潜力 0-100


class ResourceMonitor:
    """
    M3资源监控器类
    
    提供全面的系统资源监控、分析和优化功能
    """
    
    def __init__(self, 
                 monitoring_interval: float = 1.0,
                 history_size: int = 1000,
                 enable_gpu_monitoring: bool = True):
        """
        初始化资源监控器
        
        Args:
            monitoring_interval: 监控间隔时间（秒）
            history_size: 历史数据保存数量
            enable_gpu_monitoring: 是否启用GPU监控
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring and GPU_MONITORING_AVAILABLE
        
        # 数据存储
        self.history: Dict[ResourceType, deque] = {
            ResourceType.CPU: deque(maxlen=history_size),
            ResourceType.MEMORY: deque(maxlen=history_size),
            ResourceType.DISK: deque(maxlen=history_size),
            ResourceType.NETWORK: deque(maxlen=history_size),
            ResourceType.GPU: deque(maxlen=history_size) if self.enable_gpu_monitoring else None
        }
        
        # 预警规则
        self.alert_rules: List[AlertRule] = []
        
        # 成本配置
        self.cost_config = {
            ResourceType.CPU: 0.05,  # 每小时每1%使用率的成本
            ResourceType.MEMORY: 0.02,  # 每小时每GB的成本
            ResourceType.DISK: 0.001,  # 每小时每GB的成本
            ResourceType.NETWORK: 0.001,  # 每小时每GB传输的成本
            ResourceType.GPU: 0.50  # 每小时每1%使用率的成本
        }
        
        # 控制标志
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        # 网络监控基线
        self._last_network_stats = None
        self._network_baseline = None
        
        logger.info("资源监控器初始化完成")
    
    def start_monitoring(self) -> None:
        """启动监控服务"""
        if self._monitoring:
            logger.warning("监控服务已在运行")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("资源监控服务已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控服务"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("资源监控服务已停止")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self._monitoring:
            try:
                # 收集所有资源指标
                metrics = self._collect_all_metrics()
                
                # 存储历史数据
                with self._lock:
                    for metric in metrics:
                        if self.history[metric.resource_type] is not None:
                            self.history[metric.resource_type].append(metric)
                
                # 检查预警
                self._check_alerts(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(5)  # 错误后等待5秒再继续
    
    def _collect_all_metrics(self) -> List[ResourceMetrics]:
        """收集所有资源指标"""
        metrics = []
        
        # CPU指标
        cpu_metric = self._collect_cpu_metrics()
        if cpu_metric:
            metrics.append(cpu_metric)
        
        # 内存指标
        memory_metric = self._collect_memory_metrics()
        if memory_metric:
            metrics.append(memory_metric)
        
        # 磁盘指标
        disk_metrics = self._collect_disk_metrics()
        metrics.extend(disk_metrics)
        
        # 网络指标
        network_metric = self._collect_network_metrics()
        if network_metric:
            metrics.append(network_metric)
        
        # GPU指标
        if self.enable_gpu_monitoring:
            gpu_metrics = self._collect_gpu_metrics()
            metrics.extend(gpu_metrics)
        
        return metrics
    
    def _collect_cpu_metrics(self) -> Optional[ResourceMetrics]:
        """收集CPU使用率指标"""
        try:
            # 获取CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 获取CPU频率
            cpu_freq = psutil.cpu_freq()
            current_freq = cpu_freq.current if cpu_freq else 0
            max_freq = cpu_freq.max if cpu_freq else 0
            
            # 获取CPU核心数
            cpu_count = psutil.cpu_count()
            
            return ResourceMetrics(
                timestamp=datetime.now(),
                resource_type=ResourceType.CPU,
                usage_percent=cpu_percent,
                total=cpu_count,
                used=cpu_percent * cpu_count / 100,
                available=cpu_count - (cpu_percent * cpu_count / 100),
                unit="cores",
                additional_data={
                    "frequency_mhz": current_freq,
                    "max_frequency_mhz": max_freq,
                    "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                }
            )
        except Exception as e:
            logger.error(f"收集CPU指标失败: {e}")
            return None
    
    def _collect_memory_metrics(self) -> Optional[ResourceMetrics]:
        """收集内存使用指标"""
        try:
            memory = psutil.virtual_memory()
            
            return ResourceMetrics(
                timestamp=datetime.now(),
                resource_type=ResourceType.MEMORY,
                usage_percent=memory.percent,
                total=memory.total / (1024**3),  # 转换为GB
                used=memory.used / (1024**3),    # 转换为GB
                available=memory.available / (1024**3),  # 转换为GB
                unit="GB",
                additional_data={
                    "buffers": memory.buffers / (1024**3),
                    "cached": memory.cached / (1024**3),
                    "shared": getattr(memory, 'shared', 0) / (1024**3)
                }
            )
        except Exception as e:
            logger.error(f"收集内存指标失败: {e}")
            return None
    
    def _collect_disk_metrics(self) -> List[ResourceMetrics]:
        """收集磁盘使用指标"""
        metrics = []
        
        try:
            # 获取所有磁盘分区
            disk_partitions = psutil.disk_partitions()
            
            for partition in disk_partitions:
                try:
                    # 获取磁盘使用情况
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    
                    # 获取磁盘IO统计
                    disk_io = psutil.disk_io_counters(perdisk=True)
                    partition_io = disk_io.get(partition.device, None)
                    
                    metric = ResourceMetrics(
                        timestamp=datetime.now(),
                        resource_type=ResourceType.DISK,
                        usage_percent=(disk_usage.used / disk_usage.total) * 100,
                        total=disk_usage.total / (1024**3),  # 转换为GB
                        used=disk_usage.used / (1024**3),    # 转换为GB
                        available=disk_usage.free / (1024**3),  # 转换为GB
                        unit="GB",
                        additional_data={
                            "mountpoint": partition.mountpoint,
                            "filesystem": partition.fstype,
                            "read_bytes": partition_io.read_bytes if partition_io else 0,
                            "write_bytes": partition_io.write_bytes if partition_io else 0
                        }
                    )
                    metrics.append(metric)
                    
                except PermissionError:
                    # 某些系统分区可能没有权限访问
                    continue
                    
        except Exception as e:
            logger.error(f"收集磁盘指标失败: {e}")
        
        return metrics
    
    def _collect_network_metrics(self) -> Optional[ResourceMetrics]:
        """收集网络带宽指标"""
        try:
            # 获取网络IO统计
            network_io = psutil.net_io_counters()
            
            if self._last_network_stats is None:
                # 第一次收集，设置为基线
                self._last_network_stats = network_io
                return ResourceMetrics(
                    timestamp=datetime.now(),
                    resource_type=ResourceType.NETWORK,
                    usage_percent=0.0,
                    total=0,
                    used=0,
                    available=0,
                    unit="bytes/s",
                    additional_data={
                        "bytes_sent": network_io.bytes_sent,
                        "bytes_recv": network_io.bytes_recv,
                        "packets_sent": network_io.packets_sent,
                        "packets_recv": network_io.packets_recv
                    }
                )
            
            # 计算网络使用率
            time_delta = 1  # 假设间隔1秒
            
            bytes_sent_rate = (network_io.bytes_sent - self._last_network_stats.bytes_sent) / time_delta
            bytes_recv_rate = (network_io.bytes_recv - self._last_network_stats.bytes_recv) / time_delta
            
            # 更新基线
            self._last_network_stats = network_io
            
            # 计算网络使用率（相对于100MB/s的基准）
            baseline_bandwidth = 100 * 1024 * 1024  # 100MB/s
            total_rate = bytes_sent_rate + bytes_recv_rate
            usage_percent = min((total_rate / baseline_bandwidth) * 100, 100)
            
            return ResourceMetrics(
                timestamp=datetime.now(),
                resource_type=ResourceType.NETWORK,
                usage_percent=usage_percent,
                total=baseline_bandwidth,
                used=total_rate,
                available=baseline_bandwidth - total_rate,
                unit="bytes/s",
                additional_data={
                    "bytes_sent_rate": bytes_sent_rate,
                    "bytes_recv_rate": bytes_recv_rate,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv,
                    "errin": network_io.errin,
                    "errout": network_io.errout,
                    "dropin": network_io.dropin,
                    "dropout": network_io.dropout
                }
            )
            
        except Exception as e:
            logger.error(f"收集网络指标失败: {e}")
            return None
    
    def _collect_gpu_metrics(self) -> List[ResourceMetrics]:
        """收集GPU使用指标"""
        metrics = []
        
        if not self.enable_gpu_monitoring:
            return metrics
        
        try:
            if not GPU_MONITORING_AVAILABLE:
                return metrics
            
            # 获取GPU列表
            gpus = GPUtil.getGPUs()
            
            for i, gpu in enumerate(gpus):
                metric = ResourceMetrics(
                    timestamp=datetime.now(),
                    resource_type=ResourceType.GPU,
                    usage_percent=gpu.load * 100,  # GPUtil返回0-1的比例
                    total=gpu.memoryTotal,
                    used=gpu.memoryUsed,
                    available=gpu.memoryFree,
                    unit="MB",
                    additional_data={
                        "gpu_id": gpu.id,
                        "gpu_name": gpu.name,
                        "temperature": gpu.temperature,
                        "power_draw": getattr(gpu, 'powerDraw', 0),
                        "power_limit": getattr(gpu, 'powerLimit', 0)
                    }
                )
                metrics.append(metric)
                
        except Exception as e:
            logger.error(f"收集GPU指标失败: {e}")
        
        return metrics
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """添加预警规则"""
        with self._lock:
            self.alert_rules.append(rule)
        logger.info(f"添加预警规则: {rule.resource_type.value} - {rule.threshold}% - {rule.level.value}")
    
    def remove_alert_rule(self, resource_type: ResourceType, level: AlertLevel) -> bool:
        """移除预警规则"""
        with self._lock:
            for i, rule in enumerate(self.alert_rules):
                if rule.resource_type == resource_type and rule.level == level:
                    del self.alert_rules[i]
                    logger.info(f"移除预警规则: {resource_type.value} - {level.value}")
                    return True
        return False
    
    def _check_alerts(self, metrics: List[ResourceMetrics]) -> None:
        """检查预警规则"""
        current_time = datetime.now()
        
        for metric in metrics:
            for rule in self.alert_rules:
                if rule.resource_type == metric.resource_type and rule.enabled:
                    self._evaluate_alert_rule(rule, metric, current_time)
    
    def _evaluate_alert_rule(self, rule: AlertRule, metric: ResourceMetrics, current_time: datetime) -> None:
        """评估预警规则"""
        if metric.usage_percent >= rule.threshold:
            if rule.duration_seconds == 0:
                # 立即触发
                self._trigger_alert(rule, metric, current_time)
            else:
                # 需要持续时间
                self._check_duration_alert(rule, metric, current_time)
    
    def _trigger_alert(self, rule: AlertRule, metric: ResourceMetrics, current_time: datetime) -> None:
        """触发预警"""
        alert_message = (
            f"[{rule.level.value.upper()}] {metric.resource_type.value.upper()} 使用率预警: "
            f"{metric.usage_percent:.2f}% (阈值: {rule.threshold}%)"
        )
        
        logger.log(
            logging.CRITICAL if rule.level == AlertLevel.EMERGENCY else
            logging.ERROR if rule.level == AlertLevel.CRITICAL else
            logging.WARNING if rule.level == AlertLevel.WARNING else
            logging.INFO,
            alert_message
        )
        
        # 调用回调函数
        if rule.callback:
            try:
                rule.callback(rule, metric)
            except Exception as e:
                logger.error(f"预警回调函数执行失败: {e}")
    
    def _check_duration_alert(self, rule: AlertRule, metric: ResourceMetrics, current_time: datetime) -> None:
        """检查持续时间预警"""
        # 这里简化实现，实际应该跟踪每个规则的状态
        # 为了演示，我们假设如果当前指标超过阈值就触发
        if metric.usage_percent >= rule.threshold:
            self._trigger_alert(rule, metric, current_time)
    
    def get_current_metrics(self) -> Dict[ResourceType, List[ResourceMetrics]]:
        """获取当前所有资源指标"""
        with self._lock:
            return {k: list(v) if v is not None else [] for k, v in self.history.items()}
    
    def get_resource_summary(self, resource_type: ResourceType) -> Dict[str, Any]:
        """获取资源使用摘要"""
        with self._lock:
            if self.history[resource_type] is None or not self.history[resource_type]:
                return {"error": "没有可用的历史数据"}
            
            data = list(self.history[resource_type])
            if not data:
                return {"error": "没有可用的历史数据"}
            
            # 计算统计信息
            usage_values = [m.usage_percent for m in data]
            latest = data[-1]
            
            summary = {
                "resource_type": resource_type.value,
                "latest_usage": latest.usage_percent,
                "latest_timestamp": latest.timestamp.isoformat(),
                "total_resources": latest.total,
                "used_resources": latest.used,
                "available_resources": latest.available,
                "statistics": {
                    "min_usage": min(usage_values),
                    "max_usage": max(usage_values),
                    "avg_usage": statistics.mean(usage_values),
                    "median_usage": statistics.median(usage_values),
                    "std_dev": statistics.stdev(usage_values) if len(usage_values) > 1 else 0
                },
                "data_points": len(data),
                "unit": latest.unit,
                "additional_data": latest.additional_data
            }
            
            return summary
    
    def analyze_costs(self) -> List[CostAnalysis]:
        """分析资源成本"""
        cost_analysis = []
        
        for resource_type in ResourceType:
            if resource_type == ResourceType.GPU and not self.enable_gpu_monitoring:
                continue
            
            summary = self.get_resource_summary(resource_type)
            if "error" in summary:
                continue
            
            # 计算成本
            avg_usage = summary["statistics"]["avg_usage"]
            hourly_cost = 0
            
            if resource_type == ResourceType.CPU:
                hourly_cost = (avg_usage / 100) * self.cost_config[resource_type]
            elif resource_type == ResourceType.MEMORY:
                hourly_cost = summary["used_resources"] * self.cost_config[resource_type]
            elif resource_type == ResourceType.DISK:
                hourly_cost = summary["used_resources"] * self.cost_config[resource_type]
            elif resource_type == ResourceType.NETWORK:
                hourly_cost = summary["used_resources"] / (1024**3) * self.cost_config[resource_type]  # 转换为GB
            elif resource_type == ResourceType.GPU:
                hourly_cost = (avg_usage / 100) * self.cost_config[resource_type]
            
            daily_cost = hourly_cost * 24
            monthly_cost = daily_cost * 30
            
            # 计算效率评分
            efficiency_score = self._calculate_efficiency_score(resource_type, summary)
            
            # 计算优化潜力
            optimization_potential = self._calculate_optimization_potential(resource_type, summary)
            
            cost_analysis.append(CostAnalysis(
                resource_type=resource_type,
                hourly_cost=hourly_cost,
                daily_cost=daily_cost,
                monthly_cost=monthly_cost,
                efficiency_score=efficiency_score,
                optimization_potential=optimization_potential
            ))
        
        return cost_analysis
    
    def _calculate_efficiency_score(self, resource_type: ResourceType, summary: Dict[str, Any]) -> float:
        """计算效率评分"""
        avg_usage = summary["statistics"]["avg_usage"]
        
        # 基于使用率计算效率评分
        if resource_type in [ResourceType.CPU, ResourceType.GPU]:
            # 对于CPU和GPU，70-85%的使用率是最佳范围
            if 70 <= avg_usage <= 85:
                return 100.0
            elif avg_usage < 70:
                return max(0, 100 - (70 - avg_usage) * 2)
            else:
                return max(0, 100 - (avg_usage - 85) * 3)
        elif resource_type == ResourceType.MEMORY:
            # 对于内存，80-90%的使用率是最佳范围
            if 80 <= avg_usage <= 90:
                return 100.0
            elif avg_usage < 80:
                return max(0, 100 - (80 - avg_usage) * 1.5)
            else:
                return max(0, 100 - (avg_usage - 90) * 2)
        elif resource_type == ResourceType.DISK:
            # 对于磁盘，85-95%的使用率是可接受的
            if avg_usage <= 85:
                return 100.0
            elif avg_usage <= 95:
                return max(0, 100 - (avg_usage - 85) * 5)
            else:
                return max(0, 20 - (avg_usage - 95) * 2)
        elif resource_type == ResourceType.NETWORK:
            # 对于网络，任何使用率都可能是有意义的
            return max(0, 100 - abs(avg_usage - 50) * 2)
        
        return 50.0  # 默认评分
    
    def _calculate_optimization_potential(self, resource_type: ResourceType, summary: Dict[str, Any]) -> float:
        """计算优化潜力"""
        avg_usage = summary["statistics"]["avg_usage"]
        max_usage = summary["statistics"]["max_usage"]
        
        # 基于使用率变化和平均使用率计算优化潜力
        usage_variability = summary["statistics"]["std_dev"]
        
        if resource_type in [ResourceType.CPU, ResourceType.GPU]:
            if avg_usage > 90:
                return min(100, (avg_usage - 90) * 5 + usage_variability)
            elif avg_usage < 30:
                return min(100, (30 - avg_usage) * 2 + usage_variability)
            else:
                return usage_variability
        elif resource_type == ResourceType.MEMORY:
            if avg_usage > 85:
                return min(100, (avg_usage - 85) * 3 + usage_variability)
            elif avg_usage < 40:
                return min(100, (40 - avg_usage) * 1.5 + usage_variability)
            else:
                return usage_variability * 0.5
        elif resource_type == ResourceType.DISK:
            if avg_usage > 90:
                return min(100, (avg_usage - 90) * 10)
            else:
                return 0
        elif resource_type == ResourceType.NETWORK:
            return usage_variability
        
        return 0.0
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """获取资源优化建议"""
        recommendations = []
        
        for resource_type in ResourceType:
            if resource_type == ResourceType.GPU and not self.enable_gpu_monitoring:
                continue
            
            summary = self.get_resource_summary(resource_type)
            if "error" in summary:
                continue
            
            cost_analysis = self.analyze_costs()
            resource_cost = next((c for c in cost_analysis if c.resource_type == resource_type), None)
            
            if resource_cost:
                rec = self._generate_recommendation(resource_type, summary, resource_cost)
                if rec:
                    recommendations.append(rec)
        
        return recommendations
    
    def _generate_recommendation(self, resource_type: ResourceType, summary: Dict[str, Any], cost_analysis: CostAnalysis) -> Optional[Dict[str, Any]]:
        """生成优化建议"""
        avg_usage = summary["statistics"]["avg_usage"]
        max_usage = summary["statistics"]["max_usage"]
        efficiency_score = cost_analysis.efficiency_score
        optimization_potential = cost_analysis.optimization_potential
        
        recommendation = {
            "resource_type": resource_type.value,
            "current_status": "",
            "recommendations": [],
            "priority": "low",
            "estimated_savings": 0.0
        }
        
        if resource_type == ResourceType.CPU:
            if avg_usage > 90:
                recommendation["current_status"] = "CPU使用率过高"
                recommendation["recommendations"] = [
                    "考虑升级CPU或增加CPU核心数",
                    "优化CPU密集型应用程序",
                    "实施负载均衡",
                    "检查是否有不必要的CPU密集型进程"
                ]
                recommendation["priority"] = "high"
                recommendation["estimated_savings"] = cost_analysis.monthly_cost * 0.3
            elif avg_usage < 30:
                recommendation["current_status"] = "CPU资源利用率不足"
                recommendation["recommendations"] = [
                    "考虑减少CPU资源配置",
                    "整合应用程序以提高CPU利用率",
                    "实施资源池化"
                ]
                recommendation["priority"] = "medium"
                recommendation["estimated_savings"] = cost_analysis.monthly_cost * 0.2
            elif efficiency_score < 70:
                recommendation["current_status"] = "CPU使用效率较低"
                recommendation["recommendations"] = [
                    "优化应用程序性能",
                    "调整CPU亲和性设置",
                    "检查CPU缓存命中率"
                ]
                recommendation["priority"] = "medium"
                recommendation["estimated_savings"] = cost_analysis.monthly_cost * 0.15
        
        elif resource_type == ResourceType.MEMORY:
            if avg_usage > 85:
                recommendation["current_status"] = "内存使用率过高"
                recommendation["recommendations"] = [
                    "增加内存容量",
                    "优化内存使用应用程序",
                    "实施内存压缩",
                    "检查内存泄漏"
                ]
                recommendation["priority"] = "high"
                recommendation["estimated_savings"] = cost_analysis.monthly_cost * 0.4
            elif avg_usage < 40:
                recommendation["current_status"] = "内存资源利用率不足"
                recommendation["recommendations"] = [
                    "减少内存配置",
                    "实施内存超分配",
                    "优化内存分配策略"
                ]
                recommendation["priority"] = "medium"
                recommendation["estimated_savings"] = cost_analysis.monthly_cost * 0.25
        
        elif resource_type == ResourceType.DISK:
            if avg_usage > 90:
                recommendation["current_status"] = "磁盘空间不足"
                recommendation["recommendations"] = [
                    "清理不必要的文件",
                    "增加磁盘容量",
                    "实施磁盘压缩",
                    "移动数据到云存储"
                ]
                recommendation["priority"] = "critical"
                recommendation["estimated_savings"] = cost_analysis.monthly_cost * 0.2
            elif avg_usage < 50:
                recommendation["current_status"] = "磁盘空间利用率不足"
                recommendation["recommendations"] = [
                    "减少磁盘配置",
                    "实施存储虚拟化",
                    "优化存储策略"
                ]
                recommendation["priority"] = "low"
                recommendation["estimated_savings"] = cost_analysis.monthly_cost * 0.1
        
        elif resource_type == ResourceType.NETWORK:
            if max_usage > 90:
                recommendation["current_status"] = "网络带宽使用峰值过高"
                recommendation["recommendations"] = [
                    "增加网络带宽",
                    "实施流量整形",
                    "优化数据传输策略",
                    "使用CDN加速"
                ]
                recommendation["priority"] = "high"
                recommendation["estimated_savings"] = cost_analysis.monthly_cost * 0.25
            elif avg_usage < 20:
                recommendation["current_status"] = "网络带宽利用率不足"
                recommendation["recommendations"] = [
                    "减少网络带宽配置",
                    "优化网络架构",
                    "实施带宽复用"
                ]
                recommendation["priority"] = "medium"
                recommendation["estimated_savings"] = cost_analysis.monthly_cost * 0.3
        
        elif resource_type == ResourceType.GPU:
            if avg_usage > 90:
                recommendation["current_status"] = "GPU使用率过高"
                recommendation["recommendations"] = [
                    "增加GPU数量",
                    "优化GPU密集型任务",
                    "实施GPU资源调度",
                    "检查GPU温度和功耗"
                ]
                recommendation["priority"] = "high"
                recommendation["estimated_savings"] = cost_analysis.monthly_cost * 0.35
            elif avg_usage < 30:
                recommendation["current_status"] = "GPU资源利用率不足"
                recommendation["recommendations"] = [
                    "减少GPU配置",
                    "整合GPU任务",
                    "实施GPU共享"
                ]
                recommendation["priority"] = "medium"
                recommendation["estimated_savings"] = cost_analysis.monthly_cost * 0.4
        
        # 如果没有特定建议，检查整体效率
        if not recommendation["recommendations"] and efficiency_score < 80:
            recommendation["current_status"] = f"{resource_type.value}使用效率有待提升"
            recommendation["recommendations"] = [
                "监控资源使用模式",
                "优化应用程序配置",
                "实施自动扩缩容",
                "定期进行性能调优"
            ]
            recommendation["priority"] = "low"
            recommendation["estimated_savings"] = cost_analysis.monthly_cost * 0.1
        
        return recommendation if recommendation["recommendations"] else None
    
    def export_metrics(self, filepath: str, resource_type: Optional[ResourceType] = None) -> bool:
        """导出指标数据到文件"""
        try:
            data = {}
            
            if resource_type:
                with self._lock:
                    if self.history[resource_type] is not None:
                        data[resource_type.value] = [asdict(metric) for metric in self.history[resource_type]]
            else:
                with self._lock:
                    for res_type, metrics in self.history.items():
                        if metrics is not None:
                            data[res_type.value] = [asdict(metric) for metric in metrics]
            
            # 转换datetime为字符串
            for res_type, metrics in data.items():
                for metric in metrics:
                    metric['timestamp'] = metric['timestamp'].isoformat()
                    metric['resource_type'] = metric['resource_type'].value
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"指标数据已导出到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"导出指标数据失败: {e}")
            return False
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """获取系统健康报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "unknown",
            "resources": {},
            "alerts": [],
            "recommendations": []
        }
        
        # 收集各资源状态
        resource_healths = []
        
        for resource_type in ResourceType:
            if resource_type == ResourceType.GPU and not self.enable_gpu_monitoring:
                continue
            
            summary = self.get_resource_summary(resource_type)
            if "error" in summary:
                continue
            
            avg_usage = summary["statistics"]["avg_usage"]
            
            # 评估资源健康状态
            if resource_type in [ResourceType.CPU, ResourceType.GPU]:
                if avg_usage < 70:
                    health = "excellent"
                elif avg_usage < 85:
                    health = "good"
                elif avg_usage < 95:
                    health = "warning"
                else:
                    health = "critical"
            elif resource_type == ResourceType.MEMORY:
                if avg_usage < 80:
                    health = "excellent"
                elif avg_usage < 90:
                    health = "good"
                elif avg_usage < 95:
                    health = "warning"
                else:
                    health = "critical"
            elif resource_type == ResourceType.DISK:
                if avg_usage < 80:
                    health = "excellent"
                elif avg_usage < 90:
                    health = "good"
                elif avg_usage < 95:
                    health = "warning"
                else:
                    health = "critical"
            elif resource_type == ResourceType.NETWORK:
                health = "good"  # 网络使用率波动较大，评估标准不同
            
            resource_healths.append(health)
            
            report["resources"][resource_type.value] = {
                "health": health,
                "usage_percent": avg_usage,
                "latest_usage": summary["latest_usage"],
                "unit": summary["unit"]
            }
        
        # 计算整体健康状态
        if resource_healths:
            if all(h == "excellent" for h in resource_healths):
                report["overall_health"] = "excellent"
            elif any(h == "critical" for h in resource_healths):
                report["overall_health"] = "critical"
            elif any(h == "warning" for h in resource_healths):
                report["overall_health"] = "warning"
            else:
                report["overall_health"] = "good"
        
        # 添加优化建议
        report["recommendations"] = self.get_optimization_recommendations()
        
        return report


def create_default_alert_rules() -> List[AlertRule]:
    """创建默认预警规则"""
    return [
        # CPU预警规则
        AlertRule(ResourceType.CPU, 90.0, AlertLevel.WARNING),
        AlertRule(ResourceType.CPU, 95.0, AlertLevel.CRITICAL),
        AlertRule(ResourceType.CPU, 99.0, AlertLevel.EMERGENCY),
        
        # 内存预警规则
        AlertRule(ResourceType.MEMORY, 85.0, AlertLevel.WARNING),
        AlertRule(ResourceType.MEMORY, 90.0, AlertLevel.CRITICAL),
        AlertRule(ResourceType.MEMORY, 95.0, AlertLevel.EMERGENCY),
        
        # 磁盘预警规则
        AlertRule(ResourceType.DISK, 80.0, AlertLevel.WARNING),
        AlertRule(ResourceType.DISK, 90.0, AlertLevel.CRITICAL),
        AlertRule(ResourceType.DISK, 95.0, AlertLevel.EMERGENCY),
        
        # 网络预警规则
        AlertRule(ResourceType.NETWORK, 80.0, AlertLevel.WARNING),
        AlertRule(ResourceType.NETWORK, 95.0, AlertLevel.CRITICAL),
        
        # GPU预警规则
        AlertRule(ResourceType.GPU, 90.0, AlertLevel.WARNING),
        AlertRule(ResourceType.GPU, 95.0, AlertLevel.CRITICAL),
        AlertRule(ResourceType.GPU, 99.0, AlertLevel.EMERGENCY),
    ]


def alert_callback(rule: AlertRule, metric: ResourceMetrics) -> None:
    """预警回调函数示例"""
    print(f"🚨 预警触发: {rule.level.value.upper()} - {metric.resource_type.value.upper()} "
          f"使用率 {metric.usage_percent:.2f}% 超过阈值 {rule.threshold}%")


def run_resource_monitor_demo():
    """运行资源监控器演示"""
    print("=== M3资源监控器演示 ===\n")
    
    # 创建监控器实例
    monitor = ResourceMonitor(monitoring_interval=2.0, enable_gpu_monitoring=True)
    
    # 添加默认预警规则
    default_rules = create_default_alert_rules()
    for rule in default_rules:
        rule.callback = alert_callback
        monitor.add_alert_rule(rule)
    
    print("1. 启动资源监控...")
    monitor.start_monitoring()
    
    try:
        # 等待一些数据收集
        print("2. 等待数据收集...")
        time.sleep(10)
        
        # 获取当前指标
        print("\n3. 获取当前资源指标...")
        for resource_type in ResourceType:
            if resource_type == ResourceType.GPU and not monitor.enable_gpu_monitoring:
                continue
            summary = monitor.get_resource_summary(resource_type)
            if "error" not in summary:
                print(f"\n{resource_type.value.upper()} 指标:")
                print(f"  当前使用率: {summary['latest_usage']:.2f}%")
                print(f"  平均使用率: {summary['statistics']['avg_usage']:.2f}%")
                print(f"  最高使用率: {summary['statistics']['max_usage']:.2f}%")
                print(f"  资源总量: {summary['total_resources']:.2f} {summary['unit']}")
        
        # 成本分析
        print("\n4. 成本分析...")
        cost_analysis = monitor.analyze_costs()
        for cost in cost_analysis:
            print(f"\n{cost.resource_type.value.upper()} 成本分析:")
            print(f"  每小时成本: ${cost.hourly_cost:.4f}")
            print(f"  每日成本: ${cost.daily_cost:.4f}")
            print(f"  每月成本: ${cost.monthly_cost:.4f}")
            print(f"  效率评分: {cost.efficiency_score:.1f}/100")
            print(f"  优化潜力: {cost.optimization_potential:.1f}%")
        
        # 优化建议
        print("\n5. 优化建议...")
        recommendations = monitor.get_optimization_recommendations()
        for rec in recommendations:
            print(f"\n{rec['resource_type'].upper()} 优化建议:")
            print(f"  当前状态: {rec['current_status']}")
            print(f"  优先级: {rec['priority']}")
            print(f"  预计节省: ${rec['estimated_savings']:.2f}/月")
            print("  建议措施:")
            for suggestion in rec['recommendations']:
                print(f"    - {suggestion}")
        
        # 系统健康报告
        print("\n6. 系统健康报告...")
        health_report = monitor.get_system_health_report()
        print(f"整体健康状态: {health_report['overall_health'].upper()}")
        print("各资源健康状态:")
        for resource, status in health_report['resources'].items():
            print(f"  {resource.upper()}: {status['health'].upper()} ({status['usage_percent']:.1f}%)")
        
        # 导出数据
        print("\n7. 导出监控数据...")
        export_path = "/tmp/resource_metrics_export.json"
        if monitor.export_metrics(export_path):
            print(f"数据已导出到: {export_path}")
        
        print("\n=== 演示完成 ===")
        
    finally:
        print("\n停止监控...")
        monitor.stop_monitoring()


if __name__ == "__main__":
    # 运行演示
    run_resource_monitor_demo()