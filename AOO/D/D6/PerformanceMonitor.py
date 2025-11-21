"""
D6性能监控器
实现全面的性能监控框架，包括指标计算、实时监控、趋势分析等功能
"""

import time
import threading
import psutil
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum
import logging
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import weakref


class MetricType(Enum):
    """性能指标类型枚举"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    AVAILABILITY = "availability"
    CUSTOM = "custom"


class AlertLevel(Enum):
    """预警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    metric_type: MetricType
    value: float
    unit: str
    component: str
    tags: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class PerformanceAlert:
    """性能预警数据类"""
    alert_id: str
    timestamp: datetime
    level: AlertLevel
    metric_type: MetricType
    current_value: float
    threshold: float
    message: str
    component: str
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


@dataclass
class PerformanceReport:
    """性能报告数据类"""
    report_id: str
    generated_at: datetime
    time_range: Tuple[datetime, datetime]
    summary: Dict[str, Any]
    metrics: List[PerformanceMetrics]
    alerts: List[PerformanceAlert]
    trends: Dict[str, Any]
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]
    charts: List[str] = None
    
    def __post_init__(self):
        if self.charts is None:
            self.charts = []


class TrendAnalyzer:
    """性能趋势分析器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1)
    
    def analyze_trend(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """分析性能指标趋势"""
        if len(metrics) < 2:
            return {"trend": "insufficient_data", "slope": 0, "confidence": 0}
        
        # 转换为时间序列数据
        df = pd.DataFrame([{
            'timestamp': m.timestamp,
            'value': m.value,
            'metric_type': m.metric_type.value
        } for m in metrics])
        
        # 计算趋势
        df['time_diff'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['time_diff'], df['value'])
        
        # 异常检测
        values = df['value'].values.reshape(-1, 1)
        outliers = self.isolation_forest.fit_predict(values)
        anomaly_count = np.sum(outliers == -1)
        
        # 趋势判断
        if abs(slope) < 1e-6:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            "trend": trend,
            "slope": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_count / len(metrics),
            "volatility": np.std(df['value']),
            "mean": np.mean(df['value']),
            "percentile_95": np.percentile(df['value'], 95),
            "percentile_5": np.percentile(df['value'], 5)
        }
    
    def detect_patterns(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """检测性能模式"""
        if len(metrics) < 24:  # 至少需要24个数据点
            return {"patterns": []}
        
        df = pd.DataFrame([{
            'timestamp': m.timestamp,
            'value': m.value,
            'hour': m.timestamp.hour,
            'day_of_week': m.timestamp.weekday()
        } for m in metrics])
        
        patterns = []
        
        # 检测周期性模式
        hourly_avg = df.groupby('hour')['value'].mean()
        if hourly_avg.std() > hourly_avg.mean() * 0.1:  # 变异系数 > 10%
            patterns.append({
                "type": "hourly_pattern",
                "description": "检测到小时级周期性变化",
                "peak_hours": hourly_avg.nlargest(3).index.tolist(),
                "low_hours": hourly_avg.nsmallest(3).index.tolist()
            })
        
        # 检测周模式
        daily_avg = df.groupby('day_of_week')['value'].mean()
        if daily_avg.std() > daily_avg.mean() * 0.1:
            patterns.append({
                "type": "weekly_pattern", 
                "description": "检测到周级周期性变化",
                "peak_days": daily_avg.nlargest(3).index.tolist(),
                "low_days": daily_avg.nsmallest(3).index.tolist()
            })
        
        return {"patterns": patterns}


class BottleneckDetector:
    """性能瓶颈检测器"""
    
    def __init__(self):
        self.thresholds = {
            MetricType.CPU_USAGE: {"warning": 70, "critical": 90},
            MetricType.MEMORY_USAGE: {"warning": 80, "critical": 95},
            MetricType.DISK_IO: {"warning": 80, "critical": 95},
            MetricType.NETWORK_IO: {"warning": 80, "critical": 95},
            MetricType.RESPONSE_TIME: {"warning": 1000, "critical": 5000},
            MetricType.ERROR_RATE: {"warning": 5, "critical": 10}
        }
    
    def detect_bottlenecks(self, metrics: List[PerformanceMetrics]) -> List[Dict[str, Any]]:
        """检测性能瓶颈"""
        bottlenecks = []
        
        # 按组件和指标类型分组
        grouped_metrics = defaultdict(list)
        for metric in metrics:
            key = (metric.component, metric.metric_type)
            grouped_metrics[key].append(metric)
        
        for (component, metric_type), metric_list in grouped_metrics.items():
            if len(metric_list) < 10:  # 数据点太少
                continue
            
            values = [m.value for m in metric_list[-20:]]  # 最近20个数据点
            avg_value = np.mean(values)
            max_value = np.max(values)
            std_value = np.std(values)
            
            # 检查是否超过阈值
            thresholds = self.thresholds.get(metric_type, {})
            warning_threshold = thresholds.get("warning", float('inf'))
            critical_threshold = thresholds.get("critical", float('inf'))
            
            if avg_value >= critical_threshold or max_value >= critical_threshold:
                bottlenecks.append({
                    "component": component,
                    "metric_type": metric_type.value,
                    "severity": "critical",
                    "avg_value": avg_value,
                    "max_value": max_value,
                    "description": f"{component}的{metric_type.value}存在严重瓶颈",
                    "impact": "high",
                    "confidence": 0.9
                })
            elif avg_value >= warning_threshold or max_value >= warning_threshold:
                bottlenecks.append({
                    "component": component,
                    "metric_type": metric_type.value,
                    "severity": "warning",
                    "avg_value": avg_value,
                    "max_value": max_value,
                    "description": f"{component}的{metric_type.value}存在性能风险",
                    "impact": "medium",
                    "confidence": 0.7
                })
            
            # 检测异常波动
            if std_value > avg_value * 0.5:  # 变异系数 > 50%
                bottlenecks.append({
                    "component": component,
                    "metric_type": metric_type.value,
                    "severity": "warning",
                    "avg_value": avg_value,
                    "volatility": std_value,
                    "description": f"{component}的{metric_type.value}波动较大",
                    "impact": "medium",
                    "confidence": 0.6
                })
        
        return bottlenecks


class OptimizationAdvisor:
    """性能优化建议器"""
    
    def __init__(self):
        self.optimization_rules = {
            MetricType.CPU_USAGE: {
                "high": [
                    "考虑升级CPU或增加CPU核心数",
                    "优化算法复杂度，减少CPU密集型操作",
                    "使用缓存减少重复计算",
                    "考虑使用异步处理"
                ],
                "medium": [
                    "检查是否有死循环或低效代码",
                    "优化数据库查询",
                    "使用连接池减少连接开销"
                ]
            },
            MetricType.MEMORY_USAGE: {
                "high": [
                    "增加内存容量",
                    "优化内存使用，减少内存泄漏",
                    "使用对象池重用对象",
                    "考虑分页或流式处理大数据"
                ],
                "medium": [
                    "及时释放不需要的对象",
                    "优化数据结构选择",
                    "使用弱引用避免循环引用"
                ]
            },
            MetricType.RESPONSE_TIME: {
                "high": [
                    "优化数据库查询和索引",
                    "使用CDN加速静态资源",
                    "实施缓存策略",
                    "考虑负载均衡"
                ],
                "medium": [
                    "减少网络请求次数",
                    "压缩响应数据",
                    "优化前端资源加载"
                ]
            }
        }
    
    def generate_recommendations(self, bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        for bottleneck in bottlenecks:
            metric_type = MetricType(bottleneck["metric_type"])
            severity = bottleneck["severity"]
            component = bottleneck["component"]
            
            rules = self.optimization_rules.get(metric_type, {})
            if severity == "critical":
                suggestions = rules.get("high", [])
            else:
                suggestions = rules.get("medium", [])
            
            for suggestion in suggestions:
                recommendations.append(f"[{component}] {suggestion}")
        
        # 添加通用建议
        if len(bottlenecks) > 3:
            recommendations.append("考虑进行全面的性能调优和架构优化")
        
        recommendations.append("定期监控性能指标，及时发现和处理问题")
        recommendations.append("建立性能基准线，持续跟踪性能变化")
        
        return recommendations[:10]  # 限制建议数量


class PerformanceMonitor:
    """D6性能监控器主类"""
    
    def __init__(self, 
                 monitoring_interval: float = 1.0,
                 data_retention_hours: int = 24,
                 alert_thresholds: Dict[MetricType, Dict[str, float]] = None):
        """
        初始化性能监控器
        
        Args:
            monitoring_interval: 监控间隔（秒）
            data_retention_hours: 数据保留时间（小时）
            alert_thresholds: 预警阈值配置
        """
        self.monitoring_interval = monitoring_interval
        self.data_retention_hours = data_retention_hours
        self.alert_thresholds = alert_thresholds or {}
        
        # 数据存储
        self.metrics_buffer = deque(maxlen=10000)
        self.alerts_buffer = deque(maxlen=1000)
        
        # 组件状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.callbacks = defaultdict(list)
        
        # 分析组件
        self.trend_analyzer = TrendAnalyzer()
        self.bottleneck_detector = BottleneckDetector()
        self.optimization_advisor = OptimizationAdvisor()
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 性能计数器
        self.performance_counters = {}
        self._setup_performance_counters()
    
    def _setup_performance_counters(self):
        """设置性能计数器"""
        self.performance_counters = {
            'total_metrics_collected': 0,
            'total_alerts_generated': 0,
            'monitoring_uptime': 0,
            'last_collection_time': None
        }
    
    def start_monitoring(self):
        """开始性能监控"""
        if self.is_monitoring:
            self.logger.warning("监控已经在运行中")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("性能监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        start_time = time.time()
        
        while self.is_monitoring:
            try:
                # 收集系统性能指标
                self._collect_system_metrics()
                
                # 清理过期数据
                self._cleanup_old_data()
                
                # 检查预警条件
                self._check_alerts()
                
                # 更新计数器
                self.performance_counters['monitoring_uptime'] = time.time() - start_time
                self.performance_counters['last_collection_time'] = datetime.now()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(1)  # 出错时等待1秒后重试
    
    def _collect_system_metrics(self):
        """收集系统性能指标"""
        timestamp = datetime.now()
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=None)
        self._add_metric(PerformanceMetrics(
            timestamp=timestamp,
            metric_type=MetricType.CPU_USAGE,
            value=cpu_percent,
            unit="%",
            component="system"
        ))
        
        # 内存使用率
        memory = psutil.virtual_memory()
        self._add_metric(PerformanceMetrics(
            timestamp=timestamp,
            metric_type=MetricType.MEMORY_USAGE,
            value=memory.percent,
            unit="%",
            component="system"
        ))
        
        # 磁盘I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            disk_usage = psutil.disk_usage('/')
            self._add_metric(PerformanceMetrics(
                timestamp=timestamp,
                metric_type=MetricType.DISK_IO,
                value=disk_usage.percent,
                unit="%",
                component="system"
            ))
        
        # 网络I/O
        network_io = psutil.net_io_counters()
        if network_io:
            self._add_metric(PerformanceMetrics(
                timestamp=timestamp,
                metric_type=MetricType.NETWORK_IO,
                value=network_io.bytes_recv + network_io.bytes_sent,
                unit="bytes",
                component="system"
            ))
    
    def _add_metric(self, metric: PerformanceMetrics):
        """添加性能指标"""
        self.metrics_buffer.append(metric)
        self.performance_counters['total_metrics_collected'] += 1
        
        # 触发回调
        for callback in self.callbacks[metric.metric_type]:
            try:
                callback(metric)
            except Exception as e:
                self.logger.error(f"回调执行错误: {e}")
    
    def _cleanup_old_data(self):
        """清理过期数据"""
        cutoff_time = datetime.now() - timedelta(hours=self.data_retention_hours)
        
        # 清理过期指标
        while self.metrics_buffer and self.metrics_buffer[0].timestamp < cutoff_time:
            self.metrics_buffer.popleft()
        
        # 清理过期预警
        while self.alerts_buffer and self.alerts_buffer[0].timestamp < cutoff_time:
            self.alerts_buffer.popleft()
    
    def _check_alerts(self):
        """检查预警条件"""
        if len(self.metrics_buffer) < 10:
            return
        
        # 获取最新的指标
        recent_metrics = list(self.metrics_buffer)[-10:]
        
        for metric in recent_metrics:
            threshold = self.alert_thresholds.get(metric.metric_type, {})
            warning_threshold = threshold.get("warning")
            critical_threshold = threshold.get("critical")
            
            if critical_threshold and metric.value >= critical_threshold:
                self._create_alert(metric, critical_threshold, AlertLevel.CRITICAL)
            elif warning_threshold and metric.value >= warning_threshold:
                self._create_alert(metric, warning_threshold, AlertLevel.WARNING)
    
    def _create_alert(self, metric: PerformanceMetrics, threshold: float, level: AlertLevel):
        """创建预警"""
        alert_id = f"{metric.metric_type.value}_{metric.component}_{int(time.time())}"
        
        # 生成建议
        suggestions = self._generate_alert_suggestions(metric, level)
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            timestamp=metric.timestamp,
            level=level,
            metric_type=metric.metric_type,
            current_value=metric.value,
            threshold=threshold,
            message=f"{metric.component}的{metric.metric_type.value}达到{level.value}级别: {metric.value:.2f}",
            component=metric.component,
            suggestions=suggestions
        )
        
        self.alerts_buffer.append(alert)
        self.performance_counters['total_alerts_generated'] += 1
        
        self.logger.warning(f"性能预警: {alert.message}")
    
    def _generate_alert_suggestions(self, metric: PerformanceMetrics, level: AlertLevel) -> List[str]:
        """生成预警建议"""
        suggestions = []
        
        if metric.metric_type == MetricType.CPU_USAGE:
            if level == AlertLevel.CRITICAL:
                suggestions = [
                    "立即检查CPU密集型进程",
                    "考虑临时扩容或负载均衡",
                    "优化算法和查询性能"
                ]
            else:
                suggestions = [
                    "监控CPU使用趋势",
                    "优化后台任务调度"
                ]
        
        elif metric.metric_type == MetricType.MEMORY_USAGE:
            if level == AlertLevel.CRITICAL:
                suggestions = [
                    "立即释放不必要的内存",
                    "检查内存泄漏",
                    "考虑重启服务"
                ]
            else:
                suggestions = [
                    "优化内存使用",
                    "检查缓存策略"
                ]
        
        return suggestions
    
    def register_callback(self, metric_type: MetricType, callback: Callable):
        """注册指标回调函数"""
        self.callbacks[metric_type].append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前性能指标"""
        if not self.metrics_buffer:
            return {}
        
        latest_metrics = {}
        for metric in reversed(self.metrics_buffer):
            if metric.metric_type not in latest_metrics:
                latest_metrics[metric.metric_type.value] = {
                    'value': metric.value,
                    'unit': metric.unit,
                    'component': metric.component,
                    'timestamp': metric.timestamp.isoformat()
                }
        
        return latest_metrics
    
    def get_historical_metrics(self, 
                              metric_type: MetricType = None,
                              component: str = None,
                              hours: int = 1) -> List[PerformanceMetrics]:
        """获取历史性能指标"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_metrics = []
        for metric in self.metrics_buffer:
            if metric.timestamp >= cutoff_time:
                if metric_type and metric.metric_type != metric_type:
                    continue
                if component and metric.component != component:
                    continue
                filtered_metrics.append(metric)
        
        return filtered_metrics
    
    def analyze_performance(self, hours: int = 1) -> Dict[str, Any]:
        """分析性能数据"""
        metrics = self.get_historical_metrics(hours=hours)
        
        if not metrics:
            return {"error": "没有足够的性能数据进行分析"}
        
        # 按指标类型分组
        grouped_metrics = defaultdict(list)
        for metric in metrics:
            grouped_metrics[metric.metric_type].append(metric)
        
        analysis_result = {
            "analysis_time": datetime.now().isoformat(),
            "time_range_hours": hours,
            "total_metrics": len(metrics),
            "metric_types": list(grouped_metrics.keys()),
            "trends": {},
            "bottlenecks": [],
            "recommendations": []
        }
        
        # 趋势分析
        for metric_type, type_metrics in grouped_metrics.items():
            if len(type_metrics) >= 5:
                trend_result = self.trend_analyzer.analyze_trend(type_metrics)
                pattern_result = self.trend_analyzer.detect_patterns(type_metrics)
                analysis_result["trends"][metric_type.value] = {
                    **trend_result,
                    **pattern_result
                }
        
        # 瓶颈检测
        bottlenecks = self.bottleneck_detector.detect_bottlenecks(metrics)
        analysis_result["bottlenecks"] = bottlenecks
        
        # 生成优化建议
        recommendations = self.optimization_advisor.generate_recommendations(bottlenecks)
        analysis_result["recommendations"] = recommendations
        
        return analysis_result
    
    def generate_report(self, hours: int = 24) -> PerformanceReport:
        """生成性能报告"""
        metrics = self.get_historical_metrics(hours=hours)
        alerts = [alert for alert in self.alerts_buffer 
                 if alert.timestamp >= datetime.now() - timedelta(hours=hours)]
        
        # 生成报告ID
        report_id = f"perf_report_{int(time.time())}"
        
        # 分析数据
        analysis = self.analyze_performance(hours=hours)
        
        # 生成图表
        charts = self._generate_performance_charts(metrics, hours)
        
        # 创建报告
        report = PerformanceReport(
            report_id=report_id,
            generated_at=datetime.now(),
            time_range=(datetime.now() - timedelta(hours=hours), datetime.now()),
            summary={
                "total_metrics": len(metrics),
                "total_alerts": len(alerts),
                "monitoring_uptime_hours": self.performance_counters['monitoring_uptime'] / 3600,
                "avg_metrics_per_hour": len(metrics) / hours if hours > 0 else 0
            },
            metrics=metrics,
            alerts=alerts,
            trends=analysis.get("trends", {}),
            bottlenecks=analysis.get("bottlenecks", []),
            recommendations=analysis.get("recommendations", []),
            charts=charts
        )
        
        return report
    
    def _generate_performance_charts(self, metrics: List[PerformanceMetrics], hours: int) -> List[str]:
        """生成性能图表"""
        if not metrics:
            return []
        
        charts = []
        
        try:
            # 创建DataFrame
            df = pd.DataFrame([{
                'timestamp': m.timestamp,
                'value': m.value,
                'metric_type': m.metric_type.value,
                'component': m.component
            } for m in metrics])
            
            # 按指标类型分组绘制图表
            for metric_type in df['metric_type'].unique():
                metric_data = df[df['metric_type'] == metric_type]
                
                plt.figure(figsize=(12, 6))
                plt.plot(metric_data['timestamp'], metric_data['value'])
                plt.title(f'{metric_type} 性能趋势 (最近{hours}小时)')
                plt.xlabel('时间')
                plt.ylabel('值')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                chart_path = f"performance_chart_{metric_type}_{int(time.time())}.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                charts.append(chart_path)
        
        except Exception as e:
            self.logger.error(f"生成图表时出错: {e}")
        
        return charts
    
    def export_data(self, filepath: str, format: str = "json", hours: int = 1):
        """导出性能数据"""
        metrics = self.get_historical_metrics(hours=hours)
        
        if format.lower() == "json":
            data = {
                "export_time": datetime.now().isoformat(),
                "time_range_hours": hours,
                "metrics": [asdict(metric) for metric in metrics]
            }
            
            # 转换datetime为字符串
            for metric_data in data["metrics"]:
                metric_data["timestamp"] = metric_data["timestamp"].isoformat()
                metric_data["metric_type"] = metric_data["metric_type"].value
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "csv":
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'metric_type', 'value', 'unit', 'component'])
                
                for metric in metrics:
                    writer.writerow([
                        metric.timestamp.isoformat(),
                        metric.metric_type.value,
                        metric.value,
                        metric.unit,
                        metric.component
                    ])
        
        self.logger.info(f"性能数据已导出到: {filepath}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total": psutil.virtual_memory().total,
            "disk_total": psutil.disk_usage('/').total,
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            "platform": psutil.WINDOWS if psutil.WINDOWS else psutil.LINUX
        }
    
    def get_monitor_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            "is_monitoring": self.is_monitoring,
            "monitoring_interval": self.monitoring_interval,
            "data_retention_hours": self.data_retention_hours,
            "total_metrics_collected": self.performance_counters['total_metrics_collected'],
            "total_alerts_generated": self.performance_counters['total_alerts_generated'],
            "monitoring_uptime_seconds": self.performance_counters['monitoring_uptime'],
            "last_collection_time": self.performance_counters['last_collection_time'].isoformat() 
                                  if self.performance_counters['last_collection_time'] else None,
            "buffer_size": len(self.metrics_buffer),
            "alerts_buffer_size": len(self.alerts_buffer),
            "registered_callbacks": sum(len(callbacks) for callbacks in self.callbacks.values())
        }


# 示例使用和测试代码
if __name__ == "__main__":
    # 创建性能监控器
    monitor = PerformanceMonitor(
        monitoring_interval=2.0,
        data_retention_hours=24,
        alert_thresholds={
            MetricType.CPU_USAGE: {"warning": 70, "critical": 90},
            MetricType.MEMORY_USAGE: {"warning": 80, "critical": 95}
        }
    )
    
    # 注册回调
    def cpu_alert_callback(metric: PerformanceMetrics):
        if metric.value > 80:
            print(f"CPU使用率过高预警: {metric.value}%")
    
    monitor.register_callback(MetricType.CPU_USAGE, cpu_alert_callback)
    
    # 启动监控
    monitor.start_monitoring()
    
    try:
        # 运行30秒监控
        print("开始性能监控...")
        time.sleep(30)
        
        # 获取当前指标
        current_metrics = monitor.get_current_metrics()
        print("当前性能指标:", json.dumps(current_metrics, indent=2, ensure_ascii=False))
        
        # 分析性能
        analysis = monitor.analyze_performance(hours=1)
        print("性能分析结果:", json.dumps(analysis, indent=2, ensure_ascii=False))
        
        # 生成报告
        report = monitor.generate_report(hours=1)
        print(f"生成了性能报告: {report.report_id}")
        
        # 导出数据
        monitor.export_data("performance_data.json", format="json", hours=1)
        
        # 获取系统信息
        system_info = monitor.get_system_info()
        print("系统信息:", json.dumps(system_info, indent=2, ensure_ascii=False))
        
        # 获取监控状态
        status = monitor.get_monitor_status()
        print("监控状态:", json.dumps(status, indent=2, ensure_ascii=False))
        
    finally:
        # 停止监控
        monitor.stop_monitoring()
        print("性能监控已停止")