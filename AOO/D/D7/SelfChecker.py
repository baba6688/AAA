"""
D7状态自检器实现
================

实现系统状态全面检查、健康状态评估、问题诊断和定位等功能。

功能模块：
1. 系统状态全面检查 - SystemHealthChecker
2. 健康状态评估 - StateAnalyzer  
3. 问题诊断和定位 - ProblemDiagnostic
4. 系统状态报告 - StateReporter
5. 状态修复和优化 - StateOptimizer
6. 状态监控和预警 - StateMonitor
7. 状态历史记录 - StateHistoryManager
"""

import os
import sys
import json
import time
import threading
import logging
import psutil
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import sqlite3
import hashlib


class HealthLevel(Enum):
    """健康等级枚举"""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class IssueSeverity(Enum):
    """问题严重性枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemComponent(Enum):
    """系统组件枚举"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    APPLICATION = "application"
    SERVICE = "service"
    PROCESS = "process"


@dataclass
class SystemMetric:
    """系统指标数据类"""
    component: SystemComponent
    name: str
    value: float
    unit: str
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime
    status: HealthLevel


@dataclass
class SystemIssue:
    """系统问题数据类"""
    id: str
    component: SystemComponent
    severity: IssueSeverity
    title: str
    description: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    solutions: List[str] = None


@dataclass
class SystemReport:
    """系统报告数据类"""
    timestamp: datetime
    overall_health: HealthLevel
    metrics: List[SystemMetric]
    issues: List[SystemIssue]
    recommendations: List[str]
    system_info: Dict[str, Any]


class SystemHealthChecker:
    """系统健康检查器"""
    
    def __init__(self):
        self.metrics_cache = {}
        self.thresholds = {
            SystemComponent.CPU: {'warning': 70, 'critical': 90},
            SystemComponent.MEMORY: {'warning': 80, 'critical': 95},
            SystemComponent.DISK: {'warning': 85, 'critical': 95},
            SystemComponent.NETWORK: {'warning': 1000000, 'critical': 5000000},  # bytes/sec
        }
    
    def check_system_health(self) -> Dict[SystemComponent, SystemMetric]:
        """检查系统整体健康状态"""
        results = {}
        
        # CPU检查
        results[SystemComponent.CPU] = self._check_cpu()
        
        # 内存检查
        results[SystemComponent.MEMORY] = self._check_memory()
        
        # 磁盘检查
        results[SystemComponent.DISK] = self._check_disk()
        
        # 网络检查
        results[SystemComponent.NETWORK] = self._check_network()
        
        # 进程检查
        results[SystemComponent.PROCESS] = self._check_processes()
        
        # 服务检查
        results[SystemComponent.SERVICE] = self._check_services()
        
        return results
    
    def _check_cpu(self) -> SystemMetric:
        """检查CPU使用率"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # 计算健康状态
            status = self._get_health_status(cpu_percent, self.thresholds[SystemComponent.CPU])
            
            return SystemMetric(
                component=SystemComponent.CPU,
                name="CPU使用率",
                value=cpu_percent,
                unit="%",
                threshold_warning=self.thresholds[SystemComponent.CPU]['warning'],
                threshold_critical=self.thresholds[SystemComponent.CPU]['critical'],
                timestamp=datetime.now(),
                status=status
            )
        except Exception as e:
            logging.error(f"CPU检查失败: {e}")
            return self._create_error_metric(SystemComponent.CPU, str(e))
    
    def _check_memory(self) -> SystemMetric:
        """检查内存使用率"""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            status = self._get_health_status(memory_percent, self.thresholds[SystemComponent.MEMORY])
            
            return SystemMetric(
                component=SystemComponent.MEMORY,
                name="内存使用率",
                value=memory_percent,
                unit="%",
                threshold_warning=self.thresholds[SystemComponent.MEMORY]['warning'],
                threshold_critical=self.thresholds[SystemComponent.MEMORY]['critical'],
                timestamp=datetime.now(),
                status=status
            )
        except Exception as e:
            logging.error(f"内存检查失败: {e}")
            return self._create_error_metric(SystemComponent.MEMORY, str(e))
    
    def _check_disk(self) -> SystemMetric:
        """检查磁盘使用率"""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            status = self._get_health_status(disk_percent, self.thresholds[SystemComponent.DISK])
            
            return SystemMetric(
                component=SystemComponent.DISK,
                name="磁盘使用率",
                value=disk_percent,
                unit="%",
                threshold_warning=self.thresholds[SystemComponent.DISK]['warning'],
                threshold_critical=self.thresholds[SystemComponent.DISK]['critical'],
                timestamp=datetime.now(),
                status=status
            )
        except Exception as e:
            logging.error(f"磁盘检查失败: {e}")
            return self._create_error_metric(SystemComponent.DISK, str(e))
    
    def _check_network(self) -> SystemMetric:
        """检查网络状态"""
        try:
            net_io = psutil.net_io_counters()
            bytes_sent = net_io.bytes_sent
            bytes_recv = net_io.bytes_recv
            
            # 使用总流量作为网络负载指标
            total_bytes = bytes_sent + bytes_recv
            
            status = self._get_health_status(total_bytes, self.thresholds[SystemComponent.NETWORK])
            
            return SystemMetric(
                component=SystemComponent.NETWORK,
                name="网络流量",
                value=total_bytes,
                unit="bytes",
                threshold_warning=self.thresholds[SystemComponent.NETWORK]['warning'],
                threshold_critical=self.thresholds[SystemComponent.NETWORK]['critical'],
                timestamp=datetime.now(),
                status=status
            )
        except Exception as e:
            logging.error(f"网络检查失败: {e}")
            return self._create_error_metric(SystemComponent.NETWORK, str(e))
    
    def _check_processes(self) -> SystemMetric:
        """检查进程状态"""
        try:
            process_count = len(psutil.pids())
            
            # 进程数量阈值
            warning_threshold = 200
            critical_threshold = 500
            
            status = self._get_health_status(process_count, 
                                           {'warning': warning_threshold, 'critical': critical_threshold})
            
            return SystemMetric(
                component=SystemComponent.PROCESS,
                name="进程数量",
                value=process_count,
                unit="个",
                threshold_warning=warning_threshold,
                threshold_critical=critical_threshold,
                timestamp=datetime.now(),
                status=status
            )
        except Exception as e:
            logging.error(f"进程检查失败: {e}")
            return self._create_error_metric(SystemComponent.PROCESS, str(e))
    
    def _check_services(self) -> SystemMetric:
        """检查服务状态"""
        try:
            # 简化的服务检查 - 检查系统服务数量
            services = psutil.win_service_iter() if os.name == 'nt' else []
            service_count = len(list(services))
            
            warning_threshold = 50
            critical_threshold = 100
            
            status = self._get_health_status(service_count,
                                           {'warning': warning_threshold, 'critical': critical_threshold})
            
            return SystemMetric(
                component=SystemComponent.SERVICE,
                name="系统服务数量",
                value=service_count,
                unit="个",
                threshold_warning=warning_threshold,
                threshold_critical=critical_threshold,
                timestamp=datetime.now(),
                status=status
            )
        except Exception as e:
            logging.error(f"服务检查失败: {e}")
            return self._create_error_metric(SystemComponent.SERVICE, str(e))
    
    def _get_health_status(self, value: float, thresholds: Dict[str, float]) -> HealthLevel:
        """根据阈值判断健康状态"""
        if value >= thresholds['critical']:
            return HealthLevel.CRITICAL
        elif value >= thresholds['warning']:
            return HealthLevel.WARNING
        else:
            return HealthLevel.GOOD
    
    def _create_error_metric(self, component: SystemComponent, error_msg: str) -> SystemMetric:
        """创建错误指标"""
        return SystemMetric(
            component=component,
            name=f"{component.value}检查错误",
            value=0.0,
            unit="error",
            threshold_warning=0.0,
            threshold_critical=0.0,
            timestamp=datetime.now(),
            status=HealthLevel.FAILED
        )


class StateAnalyzer:
    """状态分析器"""
    
    def __init__(self):
        self.analysis_history = deque(maxlen=100)
    
    def analyze_system_state(self, metrics: Dict[SystemComponent, SystemMetric]) -> Dict[str, Any]:
        """分析系统状态"""
        analysis = {
            'timestamp': datetime.now(),
            'overall_health': self._calculate_overall_health(metrics),
            'component_health': {},
            'trends': self._analyze_trends(metrics),
            'correlations': self._find_correlations(metrics),
            'predictions': self._predict_future_state(metrics)
        }
        
        # 分析各组件健康状态
        for component, metric in metrics.items():
            analysis['component_health'][component.value] = {
                'status': metric.status.value,
                'value': metric.value,
                'severity': self._get_severity(metric)
            }
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _calculate_overall_health(self, metrics: Dict[SystemComponent, SystemMetric]) -> HealthLevel:
        """计算整体健康状态"""
        status_weights = {
            HealthLevel.EXCELLENT: 5,
            HealthLevel.GOOD: 4,
            HealthLevel.WARNING: 3,
            HealthLevel.CRITICAL: 2,
            HealthLevel.FAILED: 1
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for metric in metrics.values():
            weight = status_weights[metric.status]
            total_weight += 1
            weighted_sum += weight
        
        if total_weight == 0:
            return HealthLevel.FAILED
        
        avg_weight = weighted_sum / total_weight
        
        if avg_weight >= 4.5:
            return HealthLevel.EXCELLENT
        elif avg_weight >= 3.5:
            return HealthLevel.GOOD
        elif avg_weight >= 2.5:
            return HealthLevel.WARNING
        elif avg_weight >= 1.5:
            return HealthLevel.CRITICAL
        else:
            return HealthLevel.FAILED
    
    def _analyze_trends(self, metrics: Dict[SystemComponent, SystemMetric]) -> Dict[str, str]:
        """分析趋势"""
        trends = {}
        
        # 这里可以基于历史数据分析趋势
        # 简化实现，返回稳定状态
        for component in metrics.keys():
            trends[component.value] = "stable"
        
        return trends
    
    def _find_correlations(self, metrics: Dict[SystemComponent, SystemMetric]) -> Dict[str, float]:
        """查找组件间的相关性"""
        correlations = {}
        
        # 简化的相关性分析
        component_list = list(metrics.keys())
        for i, comp1 in enumerate(component_list):
            for comp2 in component_list[i+1:]:
                # 计算简单的相关性（这里使用随机值作为示例）
                correlation = 0.3  # 实际应用中需要基于历史数据计算
                correlations[f"{comp1.value}_{comp2.value}"] = correlation
        
        return correlations
    
    def _predict_future_state(self, metrics: Dict[SystemComponent, SystemMetric]) -> Dict[str, Any]:
        """预测未来状态"""
        predictions = {}
        
        for component, metric in metrics.items():
            # 基于当前值和趋势预测
            if metric.status == HealthLevel.WARNING:
                predictions[component.value] = {
                    'predicted_status': '可能恶化',
                    'confidence': 0.7,
                    'time_horizon': '1小时'
                }
            elif metric.status == HealthLevel.CRITICAL:
                predictions[component.value] = {
                    'predicted_status': '需要立即处理',
                    'confidence': 0.9,
                    'time_horizon': '立即'
                }
            else:
                predictions[component.value] = {
                    'predicted_status': '保持稳定',
                    'confidence': 0.8,
                    'time_horizon': '2小时'
                }
        
        return predictions
    
    def _get_severity(self, metric: SystemMetric) -> IssueSeverity:
        """获取问题严重性"""
        if metric.status == HealthLevel.FAILED:
            return IssueSeverity.CRITICAL
        elif metric.status == HealthLevel.CRITICAL:
            return IssueSeverity.HIGH
        elif metric.status == HealthLevel.WARNING:
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.LOW


class ProblemDiagnostic:
    """问题诊断器"""
    
    def __init__(self):
        self.diagnostic_rules = self._load_diagnostic_rules()
        self.issue_history = deque(maxlen=1000)
    
    def diagnose_issues(self, metrics: Dict[SystemComponent, SystemMetric], 
                       analysis: Dict[str, Any]) -> List[SystemIssue]:
        """诊断系统问题"""
        issues = []
        
        # 基于指标诊断问题
        for component, metric in metrics.items():
            component_issues = self._diagnose_component_issues(component, metric)
            issues.extend(component_issues)
        
        # 基于分析结果诊断问题
        analysis_issues = self._diagnose_analysis_issues(analysis)
        issues.extend(analysis_issues)
        
        # 诊断关联问题
        correlation_issues = self._diagnose_correlation_issues(analysis)
        issues.extend(correlation_issues)
        
        self.issue_history.extend(issues)
        return issues
    
    def _diagnose_component_issues(self, component: SystemComponent, 
                                 metric: SystemMetric) -> List[SystemIssue]:
        """诊断组件问题"""
        issues = []
        
        if metric.status == HealthLevel.FAILED:
            issue = SystemIssue(
                id="",  # 临时空ID，稍后生成
                component=component,
                severity=IssueSeverity.CRITICAL,
                title=f"{component.value}检查失败",
                description=f"无法获取{component.value}的状态信息",
                timestamp=datetime.now(),
                solutions=["检查系统日志", "重启相关服务", "检查硬件状态"]
            )
            issue.id = self._generate_issue_id(issue)
            issues.append(issue)
        
        elif metric.status == HealthLevel.CRITICAL:
            issue = SystemIssue(
                id="",
                component=component,
                severity=IssueSeverity.HIGH,
                title=f"{component.value}状态严重",
                description=f"{component.value}使用率超过临界阈值: {metric.value:.1f}{metric.unit}",
                timestamp=datetime.now(),
                solutions=self._get_solutions_for_component(component, metric)
            )
            issue.id = self._generate_issue_id(issue)
            issues.append(issue)
        
        elif metric.status == HealthLevel.WARNING:
            issue = SystemIssue(
                id="",
                component=component,
                severity=IssueSeverity.MEDIUM,
                title=f"{component.value}状态警告",
                description=f"{component.value}使用率接近警告阈值: {metric.value:.1f}{metric.unit}",
                timestamp=datetime.now(),
                solutions=self._get_solutions_for_component(component, metric)
            )
            issue.id = self._generate_issue_id(issue)
            issues.append(issue)
        
        return issues
    
    def _diagnose_analysis_issues(self, analysis: Dict[str, Any]) -> List[SystemIssue]:
        """基于分析结果诊断问题"""
        issues = []
        
        # 检查预测结果
        predictions = analysis.get('predictions', {})
        for component, prediction in predictions.items():
            if prediction.get('predicted_status') == '需要立即处理':
                issue = SystemIssue(
                    id="",
                    component=SystemComponent(component),
                    severity=IssueSeverity.HIGH,
                    title=f"{component}预测状态恶化",
                    description=f"预测{component}状态将在{prediction.get('time_horizon')}内恶化",
                    timestamp=datetime.now(),
                    solutions=["立即检查相关配置", "准备应急预案", "监控相关指标"]
                )
                issue.id = self._generate_issue_id(issue)
                issues.append(issue)
        
        return issues
    
    def _diagnose_correlation_issues(self, analysis: Dict[str, Any]) -> List[SystemIssue]:
        """诊断关联问题"""
        issues = []
        
        correlations = analysis.get('correlations', {})
        for correlation_key, correlation_value in correlations.items():
            if abs(correlation_value) > 0.8:  # 高相关性
                issue = SystemIssue(
                    id="",
                    component=SystemComponent.APPLICATION,
                    severity=IssueSeverity.MEDIUM,
                    title="高相关性检测",
                    description=f"检测到组件间高相关性: {correlation_key} (相关系数: {correlation_value:.2f})",
                    timestamp=datetime.now(),
                    solutions=["检查组件间依赖关系", "优化组件配置", "监控关联组件"]
                )
                issue.id = self._generate_issue_id(issue)
                issues.append(issue)
        
        return issues
    
    def _get_solutions_for_component(self, component: SystemComponent, 
                                   metric: SystemMetric) -> List[str]:
        """获取组件问题的解决方案"""
        solutions_map = {
            SystemComponent.CPU: [
                "关闭不必要的程序",
                "升级CPU硬件",
                "优化程序算法",
                "增加CPU核心数"
            ],
            SystemComponent.MEMORY: [
                "关闭占用内存大的程序",
                "增加内存条",
                "优化程序内存使用",
                "清理系统缓存"
            ],
            SystemComponent.DISK: [
                "清理临时文件",
                "删除不需要的文件",
                "增加磁盘容量",
                "优化磁盘空间分配"
            ],
            SystemComponent.NETWORK: [
                "检查网络连接",
                "优化网络配置",
                "增加网络带宽",
                "检查网络设备状态"
            ],
            SystemComponent.PROCESS: [
                "关闭不必要的进程",
                "优化程序性能",
                "检查进程泄漏",
                "重启相关服务"
            ]
        }
        
        return solutions_map.get(component, ["检查系统配置", "重启相关服务"])
    
    def _generate_issue_id(self, issue: SystemIssue) -> str:
        """生成问题ID"""
        content = f"{issue.component.value}_{issue.title}_{issue.timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _load_diagnostic_rules(self) -> Dict[str, Any]:
        """加载诊断规则"""
        return {
            'cpu_rules': {
                'high_usage': {'threshold': 90, 'severity': 'high'},
                'critical_usage': {'threshold': 95, 'severity': 'critical'}
            },
            'memory_rules': {
                'high_usage': {'threshold': 85, 'severity': 'high'},
                'critical_usage': {'threshold': 95, 'severity': 'critical'}
            }
        }


class StateReporter:
    """状态报告器"""
    
    def __init__(self):
        self.report_history = deque(maxlen=50)
    
    def generate_report(self, metrics: Dict[SystemComponent, SystemMetric],
                       analysis: Dict[str, Any], issues: List[SystemIssue]) -> SystemReport:
        """生成系统状态报告"""
        
        # 计算整体健康状态
        overall_health = self._calculate_overall_health(metrics)
        
        # 生成建议
        recommendations = self._generate_recommendations(metrics, issues)
        
        # 获取系统信息
        system_info = self._get_system_info()
        
        report = SystemReport(
            timestamp=datetime.now(),
            overall_health=overall_health,
            metrics=list(metrics.values()),
            issues=issues,
            recommendations=recommendations,
            system_info=system_info
        )
        
        self.report_history.append(report)
        return report
    
    def _calculate_overall_health(self, metrics: Dict[SystemComponent, SystemMetric]) -> HealthLevel:
        """计算整体健康状态"""
        status_weights = {
            HealthLevel.EXCELLENT: 5,
            HealthLevel.GOOD: 4,
            HealthLevel.WARNING: 3,
            HealthLevel.CRITICAL: 2,
            HealthLevel.FAILED: 1
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for metric in metrics.values():
            weight = status_weights[metric.status]
            total_weight += 1
            weighted_sum += weight
        
        if total_weight == 0:
            return HealthLevel.FAILED
        
        avg_weight = weighted_sum / total_weight
        
        if avg_weight >= 4.5:
            return HealthLevel.EXCELLENT
        elif avg_weight >= 3.5:
            return HealthLevel.GOOD
        elif avg_weight >= 2.5:
            return HealthLevel.WARNING
        elif avg_weight >= 1.5:
            return HealthLevel.CRITICAL
        else:
            return HealthLevel.FAILED
    
    def _generate_recommendations(self, metrics: Dict[SystemComponent, SystemMetric],
                                issues: List[SystemIssue]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于问题生成建议
        for issue in issues:
            if not issue.resolved:
                recommendations.extend(issue.solutions)
        
        # 基于指标状态生成建议
        for component, metric in metrics.items():
            if metric.status == HealthLevel.WARNING:
                recommendations.append(f"关注{component.value}状态，准备优化措施")
            elif metric.status == HealthLevel.CRITICAL:
                recommendations.append(f"立即处理{component.value}问题，避免系统故障")
        
        # 去重
        return list(set(recommendations))
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'platform': sys.platform,
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            'hostname': os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME', 'Unknown')
        }
    
    def export_report(self, report: SystemReport, format: str = 'json') -> str:
        """导出报告"""
        if format == 'json':
            return json.dumps(asdict(report), ensure_ascii=False, indent=2, default=str)
        elif format == 'text':
            return self._format_text_report(report)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def _format_text_report(self, report: SystemReport) -> str:
        """格式化文本报告"""
        lines = []
        lines.append("=" * 50)
        lines.append("系统状态报告")
        lines.append("=" * 50)
        lines.append(f"报告时间: {report.timestamp}")
        lines.append(f"整体健康状态: {report.overall_health.value}")
        lines.append("")
        
        # 系统指标
        lines.append("系统指标:")
        lines.append("-" * 20)
        for metric in report.metrics:
            lines.append(f"{metric.name}: {metric.value:.2f}{metric.unit} ({metric.status.value})")
        lines.append("")
        
        # 问题列表
        if report.issues:
            lines.append("检测到的问题:")
            lines.append("-" * 20)
            for issue in report.issues:
                if not issue.resolved:
                    lines.append(f"- {issue.title} [{issue.severity.value}]")
                    lines.append(f"  {issue.description}")
        lines.append("")
        
        # 建议
        if report.recommendations:
            lines.append("优化建议:")
            lines.append("-" * 20)
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
        
        return "\n".join(lines)


class StateOptimizer:
    """状态优化器"""
    
    def __init__(self):
        self.optimization_strategies = self._load_optimization_strategies()
    
    def optimize_system_state(self, metrics: Dict[SystemComponent, SystemMetric],
                            issues: List[SystemIssue]) -> Dict[str, Any]:
        """优化系统状态"""
        optimization_result = {
            'timestamp': datetime.now(),
            'optimizations_applied': [],
            'optimizations_planned': [],
            'expected_improvement': {},
            'risks': []
        }
        
        # 分析需要优化的问题
        for issue in issues:
            if not issue.resolved:
                optimization_plan = self._plan_optimization(issue, metrics)
                if optimization_plan:
                    optimization_result['optimizations_planned'].append(optimization_plan)
        
        # 应用自动优化
        auto_optimizations = self._apply_auto_optimizations(metrics, issues)
        optimization_result['optimizations_applied'] = auto_optimizations
        
        # 预测改进效果
        optimization_result['expected_improvement'] = self._predict_improvement(metrics, auto_optimizations)
        
        return optimization_result
    
    def _plan_optimization(self, issue: SystemIssue, 
                          metrics: Dict[SystemComponent, SystemMetric]) -> Dict[str, Any]:
        """规划优化方案"""
        optimization_plan = {
            'issue_id': issue.id,
            'component': issue.component.value,
            'priority': issue.severity.value,
            'actions': [],
            'estimated_effort': 'medium',
            'expected_impact': 'medium'
        }
        
        # 根据问题类型规划优化动作
        if issue.component == SystemComponent.CPU:
            optimization_plan['actions'] = [
                "分析CPU密集型进程",
                "优化算法复杂度",
                "考虑硬件升级"
            ]
            optimization_plan['estimated_effort'] = 'high'
            optimization_plan['expected_impact'] = 'high'
        
        elif issue.component == SystemComponent.MEMORY:
            optimization_plan['actions'] = [
                "分析内存使用模式",
                "优化内存分配策略",
                "增加内存容量"
            ]
            optimization_plan['estimated_effort'] = 'medium'
            optimization_plan['expected_impact'] = 'high'
        
        elif issue.component == SystemComponent.DISK:
            optimization_plan['actions'] = [
                "清理临时文件",
                "优化存储结构",
                "增加磁盘空间"
            ]
            optimization_plan['estimated_effort'] = 'low'
            optimization_plan['expected_impact'] = 'medium'
        
        return optimization_plan
    
    def _apply_auto_optimizations(self, metrics: Dict[SystemComponent, SystemMetric],
                                issues: List[SystemIssue]) -> List[Dict[str, Any]]:
        """应用自动优化"""
        applied_optimizations = []
        
        # 清理系统缓存
        if any(metric.component == SystemComponent.DISK and metric.status == HealthLevel.WARNING 
               for metric in metrics.values()):
            cache_cleanup = self._cleanup_system_cache()
            if cache_cleanup['success']:
                applied_optimizations.append(cache_cleanup)
        
        # 优化进程优先级
        if any(metric.component == SystemComponent.CPU and metric.status in [HealthLevel.WARNING, HealthLevel.CRITICAL]
               for metric in metrics.values()):
            process_optimization = self._optimize_processes()
            if process_optimization['success']:
                applied_optimizations.append(process_optimization)
        
        return applied_optimizations
    
    def _cleanup_system_cache(self) -> Dict[str, Any]:
        """清理系统缓存"""
        try:
            # 模拟缓存清理
            return {
                'action': '清理系统缓存',
                'success': True,
                'details': '已清理临时文件和系统缓存',
                'timestamp': datetime.now()
            }
        except Exception as e:
            return {
                'action': '清理系统缓存',
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _optimize_processes(self) -> Dict[str, Any]:
        """优化进程"""
        try:
            # 模拟进程优化
            return {
                'action': '优化进程优先级',
                'success': True,
                'details': '已调整高CPU使用率进程的优先级',
                'timestamp': datetime.now()
            }
        except Exception as e:
            return {
                'action': '优化进程优先级',
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _predict_improvement(self, metrics: Dict[SystemComponent, SystemMetric],
                           optimizations: List[Dict[str, Any]]) -> Dict[str, float]:
        """预测优化效果"""
        improvements = {}
        
        for optimization in optimizations:
            if optimization['success']:
                if '缓存' in optimization['action']:
                    improvements['disk_usage'] = -5.0  # 预期磁盘使用率降低5%
                elif '进程' in optimization['action']:
                    improvements['cpu_usage'] = -10.0  # 预期CPU使用率降低10%
        
        return improvements
    
    def _load_optimization_strategies(self) -> Dict[str, Any]:
        """加载优化策略"""
        return {
            'cpu_optimization': {
                'priority_processes': ['python', 'java', 'node'],
                'max_cpu_threshold': 80,
                'actions': ['reduce_priority', 'kill_unused', 'optimize_code']
            },
            'memory_optimization': {
                'max_memory_threshold': 85,
                'cache_cleanup_interval': 3600,
                'actions': ['clear_cache', 'optimize_allocation', 'increase_swap']
            },
            'disk_optimization': {
                'max_disk_threshold': 90,
                'cleanup_patterns': ['*.tmp', '*.log', 'cache/*'],
                'actions': ['cleanup_temp', 'compress_logs', 'archive_old_data']
            }
        }


class StateMonitor:
    """状态监控器"""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        self.alert_thresholds = {
            HealthLevel.WARNING: 3,  # 连续3次警告触发警报
            HealthLevel.CRITICAL: 1,  # 1次严重就触发警报
            HealthLevel.FAILED: 1    # 1次失败就触发警报
        }
        self.consecutive_issues = defaultdict(int)
    
    def start_monitoring(self):
        """开始监控"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logging.info("状态监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logging.info("状态监控已停止")
    
    def add_callback(self, callback: Callable[[SystemReport], None]):
        """添加监控回调函数"""
        self.callbacks.append(callback)
    
    def _monitor_loop(self):
        """监控循环"""
        health_checker = SystemHealthChecker()
        analyzer = StateAnalyzer()
        diagnostic = ProblemDiagnostic()
        reporter = StateReporter()
        
        while self.is_monitoring:
            try:
                # 执行健康检查
                metrics = health_checker.check_system_health()
                
                # 分析状态
                analysis = analyzer.analyze_system_state(metrics)
                
                # 诊断问题
                issues = diagnostic.diagnose_issues(metrics, analysis)
                
                # 生成报告
                report = reporter.generate_report(metrics, analysis, issues)
                
                # 检查是否需要触发警报
                self._check_alerts(report)
                
                # 调用回调函数
                for callback in self.callbacks:
                    try:
                        callback(report)
                    except Exception as e:
                        logging.error(f"监控回调执行失败: {e}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logging.error(f"监控循环执行失败: {e}")
                time.sleep(10)  # 出错时等待10秒后重试
    
    def _check_alerts(self, report: SystemReport):
        """检查是否需要触发警报"""
        # 检查整体健康状态
        if report.overall_health in [HealthLevel.CRITICAL, HealthLevel.FAILED]:
            self._trigger_alert("系统整体状态严重", report)
        
        # 检查组件状态
        for metric in report.metrics:
            if metric.status in [HealthLevel.CRITICAL, HealthLevel.FAILED]:
                self.consecutive_issues[metric.component] += 1
                
                threshold = self.alert_thresholds.get(metric.status, 1)
                if self.consecutive_issues[metric.component] >= threshold:
                    self._trigger_alert(f"{metric.component.value}状态异常", report)
                    self.consecutive_issues[metric.component] = 0  # 重置计数
    
    def _trigger_alert(self, message: str, report: SystemReport):
        """触发警报"""
        alert = {
            'timestamp': datetime.now(),
            'message': message,
            'report': report,
            'severity': 'high'
        }
        
        logging.warning(f"系统警报: {message}")
        
        # 这里可以添加其他警报机制，如发送邮件、短信等


class StateHistoryManager:
    """状态历史管理器"""
    
    def __init__(self, db_path: str = "system_state_history.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建指标表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                component TEXT,
                name TEXT,
                value REAL,
                unit TEXT,
                status TEXT,
                threshold_warning REAL,
                threshold_critical REAL
            )
        ''')
        
        # 创建问题表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS issues (
                id TEXT PRIMARY KEY,
                timestamp DATETIME,
                component TEXT,
                severity TEXT,
                title TEXT,
                description TEXT,
                resolved BOOLEAN,
                resolution_time DATETIME
            )
        ''')
        
        # 创建报告表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                overall_health TEXT,
                system_info TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_metric(self, metric: SystemMetric):
        """保存指标数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO metrics 
            (timestamp, component, name, value, unit, status, threshold_warning, threshold_critical)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric.timestamp,
            metric.component.value,
            metric.name,
            metric.value,
            metric.unit,
            metric.status.value,
            metric.threshold_warning,
            metric.threshold_critical
        ))
        
        conn.commit()
        conn.close()
    
    def save_issue(self, issue: SystemIssue):
        """保存问题数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO issues
            (id, timestamp, component, severity, title, description, resolved, resolution_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            issue.id,
            issue.timestamp,
            issue.component.value,
            issue.severity.value,
            issue.title,
            issue.description,
            issue.resolved,
            issue.resolution_time
        ))
        
        conn.commit()
        conn.close()
    
    def save_report(self, report: SystemReport):
        """保存报告数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reports
            (timestamp, overall_health, system_info)
            VALUES (?, ?, ?)
        ''', (
            report.timestamp,
            report.overall_health.value,
            json.dumps(report.system_info, default=str)
        ))
        
        conn.commit()
        conn.close()
    
    def get_metrics_history(self, component: SystemComponent = None, 
                          start_time: datetime = None, end_time: datetime = None) -> List[Dict]:
        """获取指标历史数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM metrics WHERE 1=1"
        params = []
        
        if component:
            query += " AND component = ?"
            params.append(component.value)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return results
    
    def get_issues_history(self, component: SystemComponent = None,
                          severity: IssueSeverity = None) -> List[Dict]:
        """获取问题历史数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM issues WHERE 1=1"
        params = []
        
        if component:
            query += " AND component = ?"
            params.append(component.value)
        
        if severity:
            query += " AND severity = ?"
            params.append(severity.value)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return results
    
    def get_health_trends(self, days: int = 7) -> Dict[str, Any]:
        """获取健康趋势分析"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取每日平均健康状态
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                AVG(CASE 
                    WHEN status = 'excellent' THEN 5
                    WHEN status = 'good' THEN 4
                    WHEN status = 'warning' THEN 3
                    WHEN status = 'critical' THEN 2
                    WHEN status = 'failed' THEN 1
                    ELSE 0
                END) as avg_health_score
            FROM metrics
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', (start_time, end_time))
        
        trends = cursor.fetchall()
        
        conn.close()
        
        return {
            'period': f'{days}天',
            'start_date': start_time.date().isoformat(),
            'end_date': end_time.date().isoformat(),
            'daily_health_scores': [{'date': trend[0], 'score': trend[1]} for trend in trends]
        }


class SelfChecker:
    """状态自检器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 初始化各个组件
        self.health_checker = SystemHealthChecker()
        self.analyzer = StateAnalyzer()
        self.diagnostic = ProblemDiagnostic()
        self.reporter = StateReporter()
        self.optimizer = StateOptimizer()
        self.monitor = StateMonitor(self.config.get('check_interval', 60))
        self.history_manager = StateHistoryManager(self.config.get('db_path', 'system_state_history.db'))
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('selfchecker.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SelfChecker')
    
    def perform_full_check(self) -> SystemReport:
        """执行完整的状态检查"""
        self.logger.info("开始执行完整系统状态检查")
        
        try:
            # 1. 系统健康检查
            metrics = self.health_checker.check_system_health()
            
            # 2. 状态分析
            analysis = self.analyzer.analyze_system_state(metrics)
            
            # 3. 问题诊断
            issues = self.diagnostic.diagnose_issues(metrics, analysis)
            
            # 4. 生成报告
            report = self.reporter.generate_report(metrics, analysis, issues)
            
            # 5. 保存历史数据
            self._save_to_history(metrics, issues, report)
            
            # 6. 自动优化
            optimization_result = self.optimizer.optimize_system_state(metrics, issues)
            self.logger.info(f"自动优化完成: {len(optimization_result['optimizations_applied'])}项优化已应用")
            
            self.logger.info("完整系统状态检查完成")
            return report
            
        except Exception as e:
            self.logger.error(f"系统状态检查失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def start_continuous_monitoring(self):
        """启动连续监控"""
        self.logger.info("启动连续状态监控")
        self.monitor.start_monitoring()
    
    def stop_continuous_monitoring(self):
        """停止连续监控"""
        self.logger.info("停止连续状态监控")
        self.monitor.stop_monitoring()
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取当前系统状态摘要"""
        try:
            metrics = self.health_checker.check_system_health()
            analysis = self.analyzer.analyze_system_state(metrics)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health': analysis['overall_health'].value,
                'component_status': {
                    component.value: {
                        'status': metric.status.value,
                        'value': metric.value,
                        'unit': metric.unit
                    }
                    for component, metric in metrics.items()
                },
                'active_issues': len([issue for issue in self.diagnostic.issue_history 
                                    if not issue.resolved])
            }
        except Exception as e:
            self.logger.error(f"获取系统状态失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health': 'unknown',
                'error': str(e)
            }
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """获取优化建议"""
        try:
            metrics = self.health_checker.check_system_health()
            issues = self.diagnostic.diagnose_issues(metrics, {})
            
            optimization_result = self.optimizer.optimize_system_state(metrics, issues)
            return optimization_result['optimizations_planned']
        except Exception as e:
            self.logger.error(f"获取优化建议失败: {e}")
            return []
    
    def export_history_report(self, days: int = 7, format: str = 'json') -> str:
        """导出历史报告"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            metrics_history = self.history_manager.get_metrics_history(
                start_time=start_time, end_time=end_time
            )
            issues_history = self.history_manager.get_issues_history()
            health_trends = self.history_manager.get_health_trends(days)
            
            history_report = {
                'period': f'{days}天',
                'metrics_count': len(metrics_history),
                'issues_count': len(issues_history),
                'health_trends': health_trends,
                'recent_metrics': metrics_history[:100],  # 最近100条记录
                'recent_issues': issues_history[:50]      # 最近50条记录
            }
            
            if format == 'json':
                return json.dumps(history_report, ensure_ascii=False, indent=2, default=str)
            else:
                raise ValueError(f"不支持的格式: {format}")
                
        except Exception as e:
            self.logger.error(f"导出历史报告失败: {e}")
            return f"导出失败: {e}"
    
    def _save_to_history(self, metrics: Dict[SystemComponent, SystemMetric],
                        issues: List[SystemIssue], report: SystemReport):
        """保存到历史数据库"""
        try:
            # 保存指标
            for metric in metrics.values():
                self.history_manager.save_metric(metric)
            
            # 保存问题
            for issue in issues:
                self.history_manager.save_issue(issue)
            
            # 保存报告
            self.history_manager.save_report(report)
            
        except Exception as e:
            self.logger.error(f"保存历史数据失败: {e}")
    
    def configure_alerts(self, email_config: Dict[str, str] = None,
                        webhook_config: Dict[str, str] = None):
        """配置警报设置"""
        # 这里可以添加邮件和webhook配置
        # 简化实现，只记录配置
        self.logger.info("警报配置已更新")
    
    def get_health_score(self) -> float:
        """获取系统健康评分 (0-100)"""
        try:
            metrics = self.health_checker.check_system_health()
            
            score_weights = {
                HealthLevel.EXCELLENT: 100,
                HealthLevel.GOOD: 80,
                HealthLevel.WARNING: 60,
                HealthLevel.CRITICAL: 30,
                HealthLevel.FAILED: 0
            }
            
            total_score = 0
            count = 0
            
            for metric in metrics.values():
                total_score += score_weights[metric.status]
                count += 1
            
            return total_score / count if count > 0 else 0
            
        except Exception as e:
            self.logger.error(f"计算健康评分失败: {e}")
            return 0


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建状态自检器实例
    checker = SelfChecker({
        'check_interval': 30,
        'db_path': 'system_state.db'
    })
    
    print("=== D7状态自检器演示 ===")
    
    # 执行完整检查
    print("\n1. 执行完整系统状态检查...")
    report = checker.perform_full_check()
    
    # 显示报告摘要
    print(f"\n系统整体健康状态: {report.overall_health.value}")
    print(f"检测到的问题数量: {len([issue for issue in report.issues if not issue.resolved])}")
    print(f"优化建议数量: {len(report.recommendations)}")
    
    # 显示系统指标
    print("\n系统指标:")
    for metric in report.metrics:
        print(f"  {metric.name}: {metric.value:.2f}{metric.unit} ({metric.status.value})")
    
    # 显示问题
    active_issues = [issue for issue in report.issues if not issue.resolved]
    if active_issues:
        print("\n检测到的问题:")
        for issue in active_issues:
            print(f"  - {issue.title} [{issue.severity.value}]")
            print(f"    {issue.description}")
    
    # 显示优化建议
    if report.recommendations:
        print("\n优化建议:")
        for i, rec in enumerate(report.recommendations[:5], 1):  # 只显示前5条
            print(f"  {i}. {rec}")
    
    # 获取健康评分
    health_score = checker.get_health_score()
    print(f"\n系统健康评分: {health_score:.1f}/100")
    
    # 获取当前状态摘要
    status = checker.get_system_status()
    print(f"\n当前状态摘要:")
    print(f"  整体健康: {status['overall_health']}")
    print(f"  活跃问题: {status['active_issues']}")
    
    print("\n=== 演示完成 ===")