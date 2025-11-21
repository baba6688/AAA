"""
Z9扩展状态聚合器主要实现

该模块实现了完整的扩展状态聚合系统，包括状态收集、数据聚合、
状态分析、报告生成、状态监控、预警机制、历史记录和仪表板功能。

主要类：
- ExtensionStatusAggregator: 主聚合器类
- StatusCollector: 状态收集器
- DataAggregator: 数据聚合器
- StatusAnalyzer: 状态分析器
- ReportGenerator: 报告生成器
- StatusMonitor: 状态监控器
- AlertManager: 预警管理器
- HistoryManager: 历史记录管理器
- DashboardManager: 仪表板管理器
"""

import json
import time
import threading
import logging
import sqlite3
import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import os


@dataclass
class ExtensionStatus:
    """扩展状态数据类"""
    extension_id: str
    name: str
    version: str
    status: str  # active, inactive, error, warning
    health_score: float  # 0.0 - 1.0
    last_update: datetime.datetime
    performance_metrics: Dict[str, float]
    error_count: int
    warning_count: int
    uptime: float
    resource_usage: Dict[str, float]
    dependencies: List[str]
    metadata: Dict[str, Any]


@dataclass
class Alert:
    """预警数据类"""
    alert_id: str
    extension_id: str
    severity: str  # low, medium, high, critical
    message: str
    timestamp: datetime.datetime
    resolved: bool = False
    resolved_at: Optional[datetime.datetime] = None


class StatusCollector:
    """状态收集器 - 从各个扩展模块收集状态信息"""
    
    def __init__(self, collector_config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = collector_config or {}
        self._collectors = {}
        self._setup_default_collectors()
    
    def _setup_default_collectors(self):
        """设置默认收集器"""
        self.register_collector("mock", self._mock_collector)
        self.register_collector("file", self._file_collector)
        self.register_collector("api", self._api_collector)
    
    def register_collector(self, name: str, collector_func: Callable):
        """注册状态收集器"""
        self._collectors[name] = collector_func
        self.logger.info(f"注册状态收集器: {name}")
    
    def collect_status(self, extension_ids: List[str], 
                      collector_type: str = "mock") -> List[ExtensionStatus]:
        """收集扩展状态"""
        if collector_type not in self._collectors:
            raise ValueError(f"未知的收集器类型: {collector_type}")
        
        collector_func = self._collectors[collector_type]
        statuses = []
        
        for ext_id in extension_ids:
            try:
                status = collector_func(ext_id)
                if status:
                    statuses.append(status)
            except Exception as e:
                self.logger.error(f"收集扩展 {ext_id} 状态失败: {e}")
        
        return statuses
    
    def _mock_collector(self, extension_id: str) -> Optional[ExtensionStatus]:
        """模拟状态收集器"""
        import random
        
        statuses = ["active", "inactive", "error", "warning"]
        status = random.choice(statuses)
        
        return ExtensionStatus(
            extension_id=extension_id,
            name=f"Extension_{extension_id}",
            version=f"1.{random.randint(0, 9)}.{random.randint(0, 9)}",
            status=status,
            health_score=random.uniform(0.3, 1.0),
            last_update=datetime.datetime.now(),
            performance_metrics={
                "cpu_usage": random.uniform(10, 90),
                "memory_usage": random.uniform(20, 80),
                "response_time": random.uniform(10, 500),
                "throughput": random.uniform(100, 1000)
            },
            error_count=random.randint(0, 5),
            warning_count=random.randint(0, 10),
            uptime=random.uniform(3600, 86400 * 30),
            resource_usage={
                "cpu": random.uniform(5, 50),
                "memory": random.uniform(10, 60),
                "disk": random.uniform(5, 30),
                "network": random.uniform(1, 20)
            },
            dependencies=[f"dep_{i}" for i in range(random.randint(0, 3))],
            metadata={
                "author": "Z9 Team",
                "category": "system",
                "tags": ["core", "essential"]
            }
        )
    
    def _file_collector(self, extension_id: str) -> Optional[ExtensionStatus]:
        """文件状态收集器"""
        # 从配置文件读取状态信息
        file_path = f"/tmp/extension_{extension_id}_status.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return ExtensionStatus(**data)
            except Exception as e:
                self.logger.error(f"读取扩展状态文件失败: {e}")
        return None
    
    def _api_collector(self, extension_id: str) -> Optional[ExtensionStatus]:
        """API状态收集器"""
        # 从API获取状态信息
        api_url = self.config.get("api_url", "")
        if api_url:
            try:
                import requests
                response = requests.get(f"{api_url}/extensions/{extension_id}/status")
                if response.status_code == 200:
                    data = response.json()
                    return ExtensionStatus(**data)
            except Exception as e:
                self.logger.error(f"API调用失败: {e}")
        return None


class DataAggregator:
    """数据聚合器 - 聚合多个扩展模块的结果"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def aggregate_statuses(self, statuses: List[ExtensionStatus]) -> Dict[str, Any]:
        """聚合状态数据"""
        if not statuses:
            return {}
        
        # 基本统计
        total_extensions = len(statuses)
        status_counts = defaultdict(int)
        health_scores = []
        error_counts = []
        warning_counts = []
        
        # 性能指标聚合
        perf_metrics = defaultdict(list)
        resource_usage = defaultdict(list)
        
        for status in statuses:
            status_counts[status.status] += 1
            health_scores.append(status.health_score)
            error_counts.append(status.error_count)
            warning_counts.append(status.warning_count)
            
            # 聚合性能指标
            for metric, value in status.performance_metrics.items():
                perf_metrics[metric].append(value)
            
            # 聚合资源使用
            for resource, usage in status.resource_usage.items():
                resource_usage[resource].append(usage)
        
        # 计算聚合结果
        aggregated = {
            "summary": {
                "total_extensions": total_extensions,
                "status_distribution": dict(status_counts),
                "average_health_score": sum(health_scores) / len(health_scores) if health_scores else 0,
                "total_errors": sum(error_counts),
                "total_warnings": sum(warning_counts),
                "active_extensions": status_counts.get("active", 0),
                "error_extensions": status_counts.get("error", 0),
                "warning_extensions": status_counts.get("warning", 0)
            },
            "performance": {
                metric: {
                    "average": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "count": len(values)
                }
                for metric, values in perf_metrics.items()
            },
            "resources": {
                resource: {
                    "average": sum(usages) / len(usages) if usages else 0,
                    "min": min(usages) if usages else 0,
                    "max": max(usages) if usages else 0,
                    "count": len(usages)
                }
                for resource, usages in resource_usage.items()
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return aggregated
    
    def aggregate_by_category(self, statuses: List[ExtensionStatus]) -> Dict[str, List[ExtensionStatus]]:
        """按类别聚合扩展"""
        categorized = defaultdict(list)
        for status in statuses:
            category = status.metadata.get("category", "uncategorized")
            categorized[category].append(status)
        return dict(categorized)
    
    def aggregate_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合趋势数据"""
        if len(historical_data) < 2:
            return {}
        
        trends = {
            "health_score_trend": [],
            "error_rate_trend": [],
            "performance_trends": defaultdict(list),
            "resource_trends": defaultdict(list)
        }
        
        for i, data in enumerate(historical_data[-10:]):  # 最近10个数据点
            summary = data.get("summary", {})
            trends["health_score_trend"].append({
                "timestamp": data.get("timestamp"),
                "value": summary.get("average_health_score", 0)
            })
            
            total_exts = summary.get("total_extensions", 1)
            error_rate = summary.get("total_errors", 0) / total_exts
            trends["error_rate_trend"].append({
                "timestamp": data.get("timestamp"),
                "value": error_rate
            })
        
        return trends


class StatusAnalyzer:
    """状态分析器 - 分析扩展状态和趋势"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._anomaly_threshold = 0.3
        self._performance_baseline = {}
    
    def analyze_status(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析状态数据"""
        analysis = {
            "health_analysis": self._analyze_health(aggregated_data),
            "performance_analysis": self._analyze_performance(aggregated_data),
            "resource_analysis": self._analyze_resources(aggregated_data),
            "risk_assessment": self._assess_risks(aggregated_data),
            "recommendations": self._generate_recommendations(aggregated_data),
            "anomaly_detection": self._detect_anomalies(aggregated_data),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return analysis
    
    def _analyze_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """健康度分析"""
        summary = data.get("summary", {})
        avg_health = summary.get("average_health_score", 0)
        
        health_status = "good"
        if avg_health < 0.5:
            health_status = "critical"
        elif avg_health < 0.7:
            health_status = "warning"
        elif avg_health < 0.85:
            health_status = "fair"
        
        return {
            "overall_health_score": avg_health,
            "health_status": health_status,
            "health_distribution": self._calculate_health_distribution(data),
            "trending": self._calculate_health_trend(data)
        }
    
    def _analyze_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """性能分析"""
        performance = data.get("performance", {})
        analysis = {}
        
        for metric, stats in performance.items():
            avg_value = stats.get("average", 0)
            analysis[metric] = {
                "average": avg_value,
                "performance_level": self._evaluate_performance_level(metric, avg_value),
                "trend": "stable",  # 简化实现
                "recommendations": self._get_performance_recommendations(metric, avg_value)
            }
        
        return analysis
    
    def _analyze_resources(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """资源分析"""
        resources = data.get("resources", {})
        analysis = {}
        
        for resource, stats in resources.items():
            avg_usage = stats.get("average", 0)
            analysis[resource] = {
                "average_usage": avg_usage,
                "usage_level": self._evaluate_resource_level(resource, avg_usage),
                "capacity_planning": self._plan_capacity(resource, avg_usage)
            }
        
        return analysis
    
    def _assess_risks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """风险评估"""
        summary = data.get("summary", {})
        risks = []
        
        # 错误率风险
        total_exts = summary.get("total_extensions", 1)
        error_rate = summary.get("total_errors", 0) / total_exts
        if error_rate > 0.1:
            risks.append({
                "type": "high_error_rate",
                "severity": "high" if error_rate > 0.2 else "medium",
                "description": f"错误率过高: {error_rate:.2%}",
                "mitigation": "检查错误日志，修复已知问题"
            })
        
        # 健康度风险
        avg_health = summary.get("average_health_score", 1.0)
        if avg_health < 0.6:
            risks.append({
                "type": "low_health_score",
                "severity": "high" if avg_health < 0.4 else "medium",
                "description": f"整体健康度偏低: {avg_health:.2f}",
                "mitigation": "优化性能，修复问题扩展"
            })
        
        return {
            "risk_level": "high" if any(r["severity"] == "high" for r in risks) else "medium",
            "risks": risks,
            "overall_score": max(0, 1 - len(risks) * 0.2)
        }
    
    def _generate_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        summary = data.get("summary", {})
        avg_health = summary.get("average_health_score", 1.0)
        
        if avg_health < 0.7:
            recommendations.append("建议对健康度较低的扩展进行性能优化")
        
        if summary.get("total_errors", 0) > 10:
            recommendations.append("错误数量较多，建议检查错误日志并修复问题")
        
        if summary.get("warning_extensions", 0) > summary.get("active_extensions", 1) * 0.3:
            recommendations.append("警告扩展比例过高，建议进行系统性检查")
        
        return recommendations
    
    def _detect_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """异常检测"""
        anomalies = []
        
        # 检测性能异常
        performance = data.get("performance", {})
        for metric, stats in performance.items():
            avg = stats.get("average", 0)
            min_val = stats.get("min", 0)
            max_val = stats.get("max", 0)
            
            if max_val - min_val > avg * 2:  # 波动过大
                anomalies.append({
                    "type": "performance_anomaly",
                    "metric": metric,
                    "description": f"{metric} 性能波动异常",
                    "severity": "medium"
                })
        
        return anomalies
    
    def _calculate_health_distribution(self, data: Dict[str, Any]) -> Dict[str, float]:
        """计算健康度分布"""
        # 简化实现
        return {"excellent": 0.4, "good": 0.3, "fair": 0.2, "poor": 0.1}
    
    def _calculate_health_trend(self, data: Dict[str, Any]) -> str:
        """计算健康度趋势"""
        # 简化实现
        return "stable"
    
    def _evaluate_performance_level(self, metric: str, value: float) -> str:
        """评估性能等级"""
        if metric == "response_time":
            if value < 100:
                return "excellent"
            elif value < 300:
                return "good"
            elif value < 500:
                return "fair"
            else:
                return "poor"
        elif metric == "throughput":
            if value > 800:
                return "excellent"
            elif value > 500:
                return "good"
            elif value > 200:
                return "fair"
            else:
                return "poor"
        return "unknown"
    
    def _get_performance_recommendations(self, metric: str, value: float) -> List[str]:
        """获取性能建议"""
        recommendations = []
        if metric == "response_time" and value > 300:
            recommendations.append("响应时间过长，建议优化代码或增加资源")
        elif metric == "throughput" and value < 300:
            recommendations.append("吞吐量较低，建议检查性能瓶颈")
        return recommendations
    
    def _evaluate_resource_level(self, resource: str, usage: float) -> str:
        """评估资源使用等级"""
        if usage > 80:
            return "critical"
        elif usage > 60:
            return "warning"
        elif usage > 40:
            return "normal"
        else:
            return "low"
    
    def _plan_capacity(self, resource: str, usage: float) -> Dict[str, Any]:
        """容量规划"""
        return {
            "current_usage": usage,
            "projected_growth": 10,  # 假设10%增长
            "capacity_needed": usage * 1.1,
            "recommendation": "monitor" if usage < 60 else "scale_up"
        }


class ReportGenerator:
    """报告生成器 - 生成综合扩展状态报告"""
    
    def __init__(self, output_dir: str = "/tmp/z9_reports"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_comprehensive_report(self, aggregated_data: Dict[str, Any], 
                                    analysis: Dict[str, Any]) -> str:
        """生成综合报告"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"extension_status_report_{timestamp}.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_markdown_report(aggregated_data, analysis))
        
        self.logger.info(f"综合报告已生成: {report_file}")
        return report_file
    
    def generate_json_report(self, aggregated_data: Dict[str, Any], 
                           analysis: Dict[str, Any]) -> str:
        """生成JSON格式报告"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"extension_status_report_{timestamp}.json")
        
        report_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "aggregated_data": aggregated_data,
            "analysis": analysis
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"JSON报告已生成: {report_file}")
        return report_file
    
    def _generate_markdown_report(self, aggregated_data: Dict[str, Any], 
                                analysis: Dict[str, Any]) -> str:
        """生成Markdown格式报告"""
        report = []
        
        # 报告标题
        report.append("# Z9扩展状态综合报告")
        report.append(f"\n**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**报告版本**: 1.0.0\n")
        
        # 执行摘要
        report.append("## 执行摘要")
        summary = aggregated_data.get("summary", {})
        report.append(f"- 总扩展数: {summary.get('total_extensions', 0)}")
        report.append(f"- 活跃扩展: {summary.get('active_extensions', 0)}")
        report.append(f"- 错误扩展: {summary.get('error_extensions', 0)}")
        report.append(f"- 平均健康度: {summary.get('average_health_score', 0):.2f}")
        report.append(f"- 总错误数: {summary.get('total_errors', 0)}")
        report.append(f"- 总警告数: {summary.get('total_warnings', 0)}\n")
        
        # 状态分布
        report.append("## 状态分布")
        status_dist = summary.get("status_distribution", {})
        for status, count in status_dist.items():
            percentage = (count / summary.get('total_extensions', 1)) * 100
            report.append(f"- {status}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # 性能分析
        report.append("## 性能分析")
        performance = aggregated_data.get("performance", {})
        for metric, stats in performance.items():
            report.append(f"### {metric}")
            report.append(f"- 平均值: {stats.get('average', 0):.2f}")
            report.append(f"- 最小值: {stats.get('min', 0):.2f}")
            report.append(f"- 最大值: {stats.get('max', 0):.2f}")
            report.append("")
        
        # 资源使用
        report.append("## 资源使用分析")
        resources = aggregated_data.get("resources", {})
        for resource, stats in resources.items():
            report.append(f"### {resource}")
            report.append(f"- 平均使用率: {stats.get('average', 0):.1f}%")
            report.append(f"- 峰值使用率: {stats.get('max', 0):.1f}%")
            report.append("")
        
        # 风险评估
        report.append("## 风险评估")
        risk_assessment = analysis.get("risk_assessment", {})
        report.append(f"**风险等级**: {risk_assessment.get('risk_level', 'unknown')}")
        report.append(f"**风险评分**: {risk_assessment.get('overall_score', 0):.2f}\n")
        
        risks = risk_assessment.get("risks", [])
        if risks:
            report.append("### 发现的风险")
            for risk in risks:
                report.append(f"- **{risk['type']}** ({risk['severity']}): {risk['description']}")
                report.append(f"  - 缓解措施: {risk['mitigation']}")
            report.append("")
        
        # 建议
        report.append("## 优化建议")
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
        else:
            report.append("暂无特殊建议，系统运行良好。")
        report.append("")
        
        # 异常检测
        report.append("## 异常检测")
        anomalies = analysis.get("anomaly_detection", [])
        if anomalies:
            for anomaly in anomalies:
                report.append(f"- **{anomaly['type']}**: {anomaly['description']}")
        else:
            report.append("未检测到异常。")
        report.append("")
        
        # 附录
        report.append("## 附录")
        report.append("### 数据收集信息")
        report.append(f"- 收集时间: {aggregated_data.get('timestamp', 'unknown')}")
        report.append(f"- 数据源: Z9扩展状态聚合器")
        report.append(f"- 分析工具: Z9状态分析引擎")
        
        return "\n".join(report)


class StatusMonitor:
    """状态监控器 - 实时监控扩展状态"""
    
    def __init__(self, monitor_interval: int = 60):
        self.monitor_interval = monitor_interval
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._monitor_thread = None
        self._status_callbacks = []
        self._alert_callbacks = []
    
    def start_monitoring(self, aggregator: 'ExtensionStatusAggregator'):
        """开始监控"""
        if self._running:
            self.logger.warning("监控已在运行中")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(aggregator,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("状态监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("状态监控已停止")
    
    def register_status_callback(self, callback: Callable):
        """注册状态回调"""
        self._status_callbacks.append(callback)
    
    def register_alert_callback(self, callback: Callable):
        """注册预警回调"""
        self._alert_callbacks.append(callback)
    
    def _monitor_loop(self, aggregator: 'ExtensionStatusAggregator'):
        """监控循环"""
        while self._running:
            try:
                # 获取当前状态
                current_status = aggregator.get_current_status()
                
                # 检查状态变化
                self._check_status_changes(current_status)
                
                # 检查预警条件
                self._check_alert_conditions(current_status)
                
                # 等待下次检查
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环出错: {e}")
                time.sleep(5)  # 出错时短暂等待
    
    def _check_status_changes(self, current_status: Dict[str, Any]):
        """检查状态变化"""
        for callback in self._status_callbacks:
            try:
                callback(current_status)
            except Exception as e:
                self.logger.error(f"状态回调执行失败: {e}")
    
    def _check_alert_conditions(self, current_status: Dict[str, Any]):
        """检查预警条件"""
        summary = current_status.get("summary", {})
        
        # 检查健康度
        avg_health = summary.get("average_health_score", 1.0)
        if avg_health < 0.5:
            self._trigger_alert("critical", f"整体健康度过低: {avg_health:.2f}")
        
        # 检查错误率
        total_exts = summary.get("total_extensions", 1)
        error_rate = summary.get("total_errors", 0) / total_exts
        if error_rate > 0.2:
            self._trigger_alert("high", f"错误率过高: {error_rate:.2%}")
    
    def _trigger_alert(self, severity: str, message: str):
        """触发预警"""
        for callback in self._alert_callbacks:
            try:
                callback(severity, message)
            except Exception as e:
                self.logger.error(f"预警回调执行失败: {e}")


class AlertManager:
    """预警管理器 - 扩展异常时预警"""
    
    def __init__(self, alert_config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = alert_config or {}
        self._alerts = {}
        self._alert_handlers = []
        self._alert_queue = queue.Queue()
    
    def create_alert(self, extension_id: str, severity: str, message: str) -> Alert:
        """创建预警"""
        alert_id = f"{extension_id}_{int(time.time())}"
        alert = Alert(
            alert_id=alert_id,
            extension_id=extension_id,
            severity=severity,
            message=message,
            timestamp=datetime.datetime.now()
        )
        
        self._alerts[alert_id] = alert
        self._alert_queue.put(alert)
        
        self.logger.warning(f"创建预警 [{severity}]: {message}")
        
        # 触发预警处理器
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"预警处理器执行失败: {e}")
        
        return alert
    
    def resolve_alert(self, alert_id: str):
        """解决预警"""
        if alert_id in self._alerts:
            alert = self._alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.datetime.now()
            self.logger.info(f"预警已解决: {alert_id}")
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃预警"""
        return [alert for alert in self._alerts.values() if not alert.resolved]
    
    def get_alerts_by_extension(self, extension_id: str) -> List[Alert]:
        """获取指定扩展的预警"""
        return [alert for alert in self._alerts.values() 
                if alert.extension_id == extension_id]
    
    def register_alert_handler(self, handler: Callable[[Alert], None]):
        """注册预警处理器"""
        self._alert_handlers.append(handler)
    
    def process_alert_queue(self):
        """处理预警队列"""
        while not self._alert_queue.empty():
            try:
                alert = self._alert_queue.get_nowait()
                self._process_alert(alert)
            except queue.Empty:
                break
    
    def _process_alert(self, alert: Alert):
        """处理单个预警"""
        # 根据严重级别采取不同处理策略
        if alert.severity == "critical":
            self._handle_critical_alert(alert)
        elif alert.severity == "high":
            self._handle_high_alert(alert)
        elif alert.severity in ["medium", "low"]:
            self._handle_normal_alert(alert)
    
    def _handle_critical_alert(self, alert: Alert):
        """处理严重预警"""
        self.logger.critical(f"严重预警: {alert.message}")
        # 可以添加邮件通知、短信通知等
    
    def _handle_high_alert(self, alert: Alert):
        """处理高级预警"""
        self.logger.error(f"高级预警: {alert.message}")
    
    def _handle_normal_alert(self, alert: Alert):
        """处理普通预警"""
        self.logger.warning(f"预警: {alert.message}")


class HistoryManager:
    """历史记录管理器 - 保存历史扩展状态"""
    
    def __init__(self, db_path: str = "/tmp/z9_extension_history.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extension_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                extension_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                health_score REAL NOT NULL,
                error_count INTEGER DEFAULT 0,
                warning_count INTEGER DEFAULT 0,
                performance_data TEXT,
                resource_data TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS aggregated_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_extensions INTEGER,
                active_extensions INTEGER,
                error_extensions INTEGER,
                warning_extensions INTEGER,
                average_health_score REAL,
                total_errors INTEGER,
                total_warnings INTEGER,
                aggregated_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_extension_status(self, status: ExtensionStatus):
        """保存扩展状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO extension_history 
            (extension_id, timestamp, status, health_score, error_count, 
             warning_count, performance_data, resource_data, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            status.extension_id,
            status.last_update.isoformat(),
            status.status,
            status.health_score,
            status.error_count,
            status.warning_count,
            json.dumps(status.performance_metrics),
            json.dumps(status.resource_usage),
            json.dumps(status.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def save_aggregated_data(self, aggregated_data: Dict[str, Any]):
        """保存聚合数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        summary = aggregated_data.get("summary", {})
        
        cursor.execute('''
            INSERT INTO aggregated_history 
            (timestamp, total_extensions, active_extensions, error_extensions,
             warning_extensions, average_health_score, total_errors, 
             total_warnings, aggregated_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            aggregated_data.get("timestamp"),
            summary.get("total_extensions"),
            summary.get("active_extensions"),
            summary.get("error_extensions"),
            summary.get("warning_extensions"),
            summary.get("average_health_score"),
            summary.get("total_errors"),
            summary.get("total_warnings"),
            json.dumps(aggregated_data)
        ))
        
        conn.commit()
        conn.close()
    
    def get_extension_history(self, extension_id: str, 
                            days: int = 7) -> List[Dict[str, Any]]:
        """获取扩展历史记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = (datetime.datetime.now() - 
                     datetime.timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT * FROM extension_history 
            WHERE extension_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        ''', (extension_id, since_date))
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            # 解析JSON字段
            if result.get('performance_data'):
                result['performance_metrics'] = json.loads(result['performance_data'])
            if result.get('resource_data'):
                result['resource_usage'] = json.loads(result['resource_data'])
            if result.get('metadata'):
                result['metadata'] = json.loads(result['metadata'])
            results.append(result)
        
        conn.close()
        return results
    
    def get_aggregated_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取聚合历史记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = (datetime.datetime.now() - 
                     datetime.timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT * FROM aggregated_history 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        ''', (since_date,))
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            if result.get('aggregated_data'):
                result['aggregated_data'] = json.loads(result['aggregated_data'])
            results.append(result)
        
        conn.close()
        return results
    
    def cleanup_old_records(self, days: int = 90):
        """清理旧记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.datetime.now() - 
                      datetime.timedelta(days=days)).isoformat()
        
        cursor.execute('DELETE FROM extension_history WHERE timestamp < ?', (cutoff_date,))
        cursor.execute('DELETE FROM aggregated_history WHERE timestamp < ?', (cutoff_date,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        self.logger.info(f"清理了 {deleted_count} 条旧记录")


class DashboardManager:
    """仪表板管理器 - 提供可视化的扩展状态仪表板"""
    
    def __init__(self, output_dir: str = "/tmp/z9_dashboard"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_dashboard(self, aggregated_data: Dict[str, Any], 
                         analysis: Dict[str, Any]) -> str:
        """生成仪表板"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_file = os.path.join(self.output_dir, f"dashboard_{timestamp}.html")
        
        html_content = self._generate_dashboard_html(aggregated_data, analysis)
        
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"仪表板已生成: {dashboard_file}")
        return dashboard_file
    
    def _generate_dashboard_html(self, aggregated_data: Dict[str, Any], 
                               analysis: Dict[str, Any]) -> str:
        """生成仪表板HTML"""
        # 简化的HTML仪表板
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Z9扩展状态仪表板</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        .status-good {{ color: #27ae60; }}
        .status-warning {{ color: #f39c12; }}
        .status-error {{ color: #e74c3c; }}
        .section {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .chart {{ width: 100%; height: 300px; margin: 20px 0; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Z9扩展状态仪表板</h1>
            <p>更新时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{aggregated_data.get('summary', {}).get('total_extensions', 0)}</div>
                <div class="metric-label">总扩展数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value status-good">{aggregated_data.get('summary', {}).get('active_extensions', 0)}</div>
                <div class="metric-label">活跃扩展</div>
            </div>
            <div class="metric-card">
                <div class="metric-value status-error">{aggregated_data.get('summary', {}).get('error_extensions', 0)}</div>
                <div class="metric-label">错误扩展</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{aggregated_data.get('summary', {}).get('average_health_score', 0):.2f}</div>
                <div class="metric-label">平均健康度</div>
            </div>
        </div>
        
        <div class="section">
            <h2>状态分布</h2>
            <canvas id="statusChart" class="chart"></canvas>
        </div>
        
        <div class="section">
            <h2>性能指标</h2>
            <canvas id="performanceChart" class="chart"></canvas>
        </div>
        
        <div class="section">
            <h2>资源使用</h2>
            <canvas id="resourceChart" class="chart"></canvas>
        </div>
        
        <div class="section">
            <h2>风险评估</h2>
            <p>风险等级: {analysis.get('risk_assessment', {}).get('risk_level', 'unknown')}</p>
            <p>风险评分: {analysis.get('risk_assessment', {}).get('overall_score', 0):.2f}</p>
        </div>
    </div>
    
    <script>
        // 状态分布图表
        const statusData = {json.dumps(aggregated_data.get('summary', {}).get('status_distribution', {}))};
        new Chart(document.getElementById('statusChart'), {{
            type: 'doughnut',
            data: {{
                labels: Object.keys(statusData),
                datasets: [{{
                    data: Object.values(statusData),
                    backgroundColor: ['#27ae60', '#e74c3c', '#f39c12', '#95a5a6']
                }}]
            }}
        }});
        
        // 性能指标图表
        const performanceData = {json.dumps({k: v.get('average', 0) for k, v in aggregated_data.get('performance', {}).items()})};
        new Chart(document.getElementById('performanceChart'), {{
            type: 'bar',
            data: {{
                labels: Object.keys(performanceData),
                datasets: [{{
                    label: '平均值',
                    data: Object.values(performanceData),
                    backgroundColor: '#3498db'
                }}]
            }}
        }});
        
        // 资源使用图表
        const resourceData = {json.dumps({k: v.get('average', 0) for k, v in aggregated_data.get('resources', {}).items()})};
        new Chart(document.getElementById('resourceChart'), {{
            type: 'line',
            data: {{
                labels: Object.keys(resourceData),
                datasets: [{{
                    label: '使用率(%)',
                    data: Object.values(resourceData),
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)'
                }}]
            }}
        }});
    </script>
</body>
</html>
        """
        return html


class ExtensionStatusAggregator:
    """Z9扩展状态聚合器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化扩展状态聚合器
        
        Args:
            config: 配置字典，包含各个组件的配置
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # 初始化各个组件
        self.status_collector = StatusCollector(self.config.get('collector', {}))
        self.data_aggregator = DataAggregator()
        self.status_analyzer = StatusAnalyzer()
        self.report_generator = ReportGenerator(self.config.get('report_dir', '/tmp/z9_reports'))
        self.status_monitor = StatusMonitor(self.config.get('monitor_interval', 60))
        self.alert_manager = AlertManager(self.config.get('alert', {}))
        self.history_manager = HistoryManager(self.config.get('db_path', '/tmp/z9_extension_history.db'))
        self.dashboard_manager = DashboardManager(self.config.get('dashboard_dir', '/tmp/z9_dashboard'))
        
        # 内部状态
        self._current_status = {}
        self._extension_ids = []
        
        self.logger.info("Z9扩展状态聚合器初始化完成")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def add_extension(self, extension_id: str):
        """添加要监控的扩展"""
        if extension_id not in self._extension_ids:
            self._extension_ids.append(extension_id)
            self.logger.info(f"添加扩展监控: {extension_id}")
    
    def remove_extension(self, extension_id: str):
        """移除监控的扩展"""
        if extension_id in self._extension_ids:
            self._extension_ids.remove(extension_id)
            self.logger.info(f"移除扩展监控: {extension_id}")
    
    def collect_status(self, collector_type: str = "mock") -> List[ExtensionStatus]:
        """收集扩展状态"""
        self.logger.info(f"开始收集 {len(self._extension_ids)} 个扩展的状态")
        
        statuses = self.status_collector.collect_status(self._extension_ids, collector_type)
        
        # 保存到历史记录
        for status in statuses:
            self.history_manager.save_extension_status(status)
        
        self.logger.info(f"状态收集完成，获得 {len(statuses)} 个状态")
        return statuses
    
    def aggregate_data(self, statuses: List[ExtensionStatus]) -> Dict[str, Any]:
        """聚合数据"""
        self.logger.info("开始数据聚合")
        aggregated = self.data_aggregator.aggregate_statuses(statuses)
        
        # 保存聚合数据
        self.history_manager.save_aggregated_data(aggregated)
        
        # 更新当前状态
        self._current_status = aggregated
        
        self.logger.info("数据聚合完成")
        return aggregated
    
    def analyze_status(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析状态"""
        self.logger.info("开始状态分析")
        analysis = self.status_analyzer.analyze_status(aggregated_data)
        self.logger.info("状态分析完成")
        return analysis
    
    def generate_report(self, aggregated_data: Dict[str, Any], 
                       analysis: Dict[str, Any]) -> Tuple[str, str]:
        """生成报告"""
        self.logger.info("开始生成报告")
        
        # 生成Markdown报告
        md_report = self.report_generator.generate_comprehensive_report(
            aggregated_data, analysis
        )
        
        # 生成JSON报告
        json_report = self.report_generator.generate_json_report(
            aggregated_data, analysis
        )
        
        self.logger.info("报告生成完成")
        return md_report, json_report
    
    def generate_dashboard(self, aggregated_data: Dict[str, Any], 
                         analysis: Dict[str, Any]) -> str:
        """生成仪表板"""
        self.logger.info("开始生成仪表板")
        dashboard = self.dashboard_manager.generate_dashboard(aggregated_data, analysis)
        self.logger.info("仪表板生成完成")
        return dashboard
    
    def start_monitoring(self):
        """开始实时监控"""
        self.logger.info("启动实时监控")
        self.status_monitor.start_monitoring(self)
    
    def stop_monitoring(self):
        """停止实时监控"""
        self.logger.info("停止实时监控")
        self.status_monitor.stop_monitoring()
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return self._current_status
    
    def get_extension_history(self, extension_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """获取扩展历史"""
        return self.history_manager.get_extension_history(extension_id, days)
    
    def get_aggregated_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取聚合历史"""
        return self.history_manager.get_aggregated_history(days)
    
    def run_full_cycle(self, collector_type: str = "mock") -> Dict[str, Any]:
        """运行完整的状态聚合周期"""
        self.logger.info("开始完整状态聚合周期")
        
        try:
            # 1. 收集状态
            statuses = self.collect_status(collector_type)
            
            # 2. 聚合数据
            aggregated_data = self.aggregate_data(statuses)
            
            # 3. 分析状态
            analysis = self.analyze_status(aggregated_data)
            
            # 4. 生成报告
            md_report, json_report = self.generate_report(aggregated_data, analysis)
            
            # 5. 生成仪表板
            dashboard = self.generate_dashboard(aggregated_data, analysis)
            
            result = {
                "statuses": [asdict(status) for status in statuses],
                "aggregated_data": aggregated_data,
                "analysis": analysis,
                "reports": {
                    "markdown": md_report,
                    "json": json_report,
                    "dashboard": dashboard
                },
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            self.logger.info("完整状态聚合周期完成")
            return result
            
        except Exception as e:
            self.logger.error(f"状态聚合周期执行失败: {e}")
            raise
    
    def cleanup_old_records(self, days: int = 90):
        """清理旧记录"""
        self.logger.info(f"清理 {days} 天前的旧记录")
        self.history_manager.cleanup_old_records(days)


def main():
    """主函数 - 演示用法"""
    # 配置聚合器
    config = {
        'collector': {
            'api_url': 'http://localhost:8080'
        },
        'monitor_interval': 30,
        'report_dir': '/tmp/z9_reports',
        'dashboard_dir': '/tmp/z9_dashboard',
        'db_path': '/tmp/z9_extension_history.db'
    }
    
    # 创建聚合器
    aggregator = ExtensionStatusAggregator(config)
    
    # 添加要监控的扩展
    extensions = [f"ext_{i}" for i in range(1, 6)]
    for ext_id in extensions:
        aggregator.add_extension(ext_id)
    
    # 运行完整周期
    result = aggregator.run_full_cycle()
    
    print("状态聚合完成!")
    print(f"Markdown报告: {result['reports']['markdown']}")
    print(f"JSON报告: {result['reports']['json']}")
    print(f"仪表板: {result['reports']['dashboard']}")
    
    # 启动实时监控
    aggregator.start_monitoring()
    
    try:
        # 保持运行一段时间
        time.sleep(10)
    finally:
        aggregator.stop_monitoring()


if __name__ == "__main__":
    main()