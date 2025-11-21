#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M9监控状态聚合器模块
===================

M9监控状态聚合器是监控层的核心组件，负责整合和协调所有监控模块的状态和数据。

主要功能：
1. 聚合M1-M8所有监控模块的状态数据
2. 提供统一的监控状态管理接口
3. 生成综合监控报告和仪表板
4. 管理跨模块的监控协调和告警
5. 提供监控数据的统一查询和分析
6. 实现监控状态的预警和异常处理
7. 支持监控配置的集中管理
8. 提供监控历史数据的存储和查询

版本: 1.0.0
创建时间: 2025-11-13
"""

import time
import json
import logging
import threading
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import copy

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AggregateStatus(Enum):
    """聚合状态枚举"""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"
    MAINTENANCE = "MAINTENANCE"


class AlertSeverity(Enum):
    """告警严重级别"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class MonitoringMetrics:
    """监控指标数据类"""
    timestamp: datetime
    source_module: str
    metric_name: str
    metric_value: float
    metric_unit: str
    status: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class AggregateAlert:
    """聚合告警数据类"""
    id: str
    timestamp: datetime
    source_modules: List[str]
    alert_type: str
    severity: AlertSeverity
    title: str
    description: str
    metrics: List[MonitoringMetrics]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class SystemOverview:
    """系统概览数据类"""
    timestamp: datetime
    overall_status: AggregateStatus
    active_modules: int
    total_alerts: int
    critical_alerts: int
    warning_alerts: int
    system_health_score: float
    module_statuses: Dict[str, str]


@dataclass
class MonitoringState:
    """监控状态数据类"""
    timestamp: datetime
    system_overview: SystemOverview
    module_states: Dict[str, Dict[str, Any]]
    active_alerts: List[AggregateAlert]
    performance_summary: Dict[str, Any]
    resource_usage: Dict[str, Any]
    recommendations: List[str]


class MonitoringStateAggregator:
    """监控状态聚合器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化监控状态聚合器
        
        Args:
            config: 配置字典，包含监控参数设置
        """
        self.config = config or {}
        self._setup_logging()
        self._setup_database()
        self._initialize_monitors()
        
        # 监控状态存储
        self._monitoring_state: Optional[MonitoringState] = None
        self._alert_history = deque(maxlen=1000)
        self._state_history = deque(maxlen=100)
        
        # 线程安全控制
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        # 执行器
        self._executor = ThreadPoolExecutor(max_workers=8)
        
        # 监控任务
        self._aggregation_task = None
        self._cleanup_task = None
        
        logger.info("M9监控状态聚合器初始化完成")
    
    def _setup_logging(self):
        """设置日志配置"""
        log_level = self.config.get('log_level', logging.INFO)
        logger.setLevel(log_level)
    
    def _setup_database(self):
        """设置数据库"""
        db_path = self.config.get('database_path', 'monitoring_aggregator.db')
        self.db_path = db_path
        
        # 创建数据库表
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 创建监控状态表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitoring_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                overall_status TEXT,
                active_modules INTEGER,
                total_alerts INTEGER,
                critical_alerts INTEGER,
                warning_alerts INTEGER,
                health_score REAL,
                state_data TEXT
            )
        ''')
        
        # 创建告警历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT,
                timestamp DATETIME,
                source_modules TEXT,
                alert_type TEXT,
                severity TEXT,
                title TEXT,
                description TEXT,
                resolved BOOLEAN,
                resolution_time DATETIME
            )
        ''')
        
        # 创建指标历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                source_module TEXT,
                metric_name TEXT,
                metric_value REAL,
                metric_unit TEXT,
                status TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _initialize_monitors(self):
        """初始化监控器实例"""
        try:
            # 尝试从不同路径导入监控模块
            import sys
            import os
            
            # 获取项目根目录路径
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            # 导入所有监控模块
            from M.M1.SystemMonitor import SystemMonitor
            from M.M2.PerformanceMonitor import PerformanceMonitor
            from M.M3.ResourceMonitor import ResourceMonitor
            from M.M4.NetworkMonitor import NetworkMonitor
            from M.M5.DataMonitor import DataMonitor
            from M.M6.TradingMonitor import TradingMonitor
            from M.M7.RiskMonitor import RiskMonitor
            from M.M8.HealthChecker import HealthChecker
            
            # 初始化监控器（使用正确的参数）
            self.monitors = {
                'M1': SystemMonitor(**self.config.get('M1', {})),
                'M2': PerformanceMonitor(**self.config.get('M2', {})),
                'M3': ResourceMonitor(**self.config.get('M3', {})),
                'M4': NetworkMonitor(**self.config.get('M4', {})),
                'M5': DataMonitor(**self.config.get('M5', {})),
                'M6': TradingMonitor(**self.config.get('M6', {})),
                'M7': RiskMonitor(**self.config.get('M7', {})),
                'M8': HealthChecker(**self.config.get('M8', {}))
            }
            
            logger.info("✓ 真实监控器初始化完成 - 使用真实监控数据")
            
        except ImportError as e:
            logger.warning(f"导入真实监控模块失败: {e}")
            logger.info("⚠️  使用模拟监控器 - 仅用于测试")
            # 如果导入失败，使用模拟监控器
            self.monitors = self._create_mock_monitors()
    
    def _create_mock_monitors(self) -> Dict[str, Any]:
        """创建模拟监控器（用于测试）"""
        class MockMonitor:
            def __init__(self, name: str):
                self.name = name
                self.last_update = datetime.now()
                
            def get_status(self) -> Dict[str, Any]:
                """提供与真实监控器兼容的接口"""
                return {
                    'status': 'RUNNING',
                    'health': 'GOOD',
                    'metrics': {
                        'cpu_usage': 45.0,
                        'memory_usage': 62.0,
                        'disk_usage': 38.0
                    },
                    'alerts': [],
                    'last_update': self.last_update
                }
                
            def get_current_metrics(self) -> Dict[str, Any]:
                """模拟真实监控器的get_current_metrics方法"""
                metrics = {
                    'cpu_usage': 45.0,
                    'memory_usage': 62.0,
                    'disk_usage': 38.0,
                    'timestamp': self.last_update
                }
                
                # 根据模块类型添加特定指标
                if self.name == 'M2':  # 性能监控器
                    metrics.update({
                        'response_time': 120.0,
                        'throughput': 500.0,
                        'error_rate': 0.02
                    })
                elif self.name == 'M6':  # 交易监控器
                    metrics.update({
                        'trades_count': 150,
                        'success_rate': 0.98,
                        'total_volume': 1000000.0
                    })
                elif self.name == 'M7':  # 风险监控器
                    metrics.update({
                        'risk_score': 0.15,
                        'position_risk': 0.12,
                        'market_risk': 0.18
                    })
                
                return metrics
                
            def get_system_health(self) -> str:
                """模拟健康状态"""
                return 'HEALTHY'
                
            def get_active_alerts(self) -> List:
                """模拟告警列表（空）"""
                return []
                
            # 适配真实监控器的其他方法
            def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
                return []
                
            def get_performance_baseline(self, metric_name: str):
                return None
                
            def get_throughput_metrics(self, hours: int = 1) -> Dict[str, float]:
                return {'throughput': 500.0}
        
        return {
            f'M{i}': MockMonitor(f'M{i}') for i in range(1, 9)
        }
    
    def start_monitoring(self):
        """启动监控聚合"""
        if self._aggregation_task and self._aggregation_task.is_alive():
            logger.warning("监控聚合已在运行")
            return
        
        # 启动聚合任务
        self._aggregation_task = threading.Thread(target=self._aggregation_loop, daemon=True)
        self._aggregation_task.start()
        
        # 启动清理任务
        self._cleanup_task = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_task.start()
        
        logger.info("监控聚合已启动")
    
    def stop_monitoring(self):
        """停止监控聚合"""
        self._stop_event.set()
        
        if self._aggregation_task:
            self._aggregation_task.join(timeout=10)
        
        if self._cleanup_task:
            self._cleanup_task.join(timeout=10)
        
        self._executor.shutdown(wait=True)
        logger.info("监控聚合已停止")
    
    def _aggregation_loop(self):
        """聚合循环"""
        while not self._stop_event.is_set():
            try:
                self._aggregate_monitoring_state()
                time.sleep(self.config.get('aggregation_interval', 30))
            except Exception as e:
                logger.error(f"聚合循环异常: {e}")
                time.sleep(60)  # 异常时延长等待时间
    
    def _cleanup_loop(self):
        """清理循环"""
        while not self._stop_event.is_set():
            try:
                self._cleanup_old_data()
                time.sleep(self.config.get('cleanup_interval', 3600))  # 每小时清理
            except Exception as e:
                logger.error(f"清理循环异常: {e}")
                time.sleep(1800)  # 异常时缩短等待时间
    
    def _aggregate_monitoring_state(self):
        """聚合监控状态"""
        with self._lock:
            try:
                start_time = time.time()
                
                # 并行获取各监控器状态
                futures = {}
                for module_name, monitor in self.monitors.items():
                    future = self._executor.submit(self._get_monitor_state, module_name, monitor)
                    futures[future] = module_name
                
                # 收集结果
                module_states = {}
                all_metrics = []
                all_alerts = []
                
                for future in as_completed(futures):
                    module_name = futures[future]
                    try:
                        state, metrics, alerts = future.result(timeout=30)
                        module_states[module_name] = state
                        all_metrics.extend(metrics)
                        all_alerts.extend(alerts)
                    except Exception as e:
                        logger.error(f"获取{module_name}状态失败: {e}")
                        module_states[module_name] = {
                            'status': 'ERROR',
                            'health': 'UNKNOWN',
                            'error': str(e)
                        }
                
                # 分析状态
                overall_status = self._analyze_overall_status(module_states, all_alerts)
                system_overview = self._create_system_overview(module_states, all_alerts)
                
                # 生成聚合状态
                self._monitoring_state = MonitoringState(
                    timestamp=datetime.now(),
                    system_overview=system_overview,
                    module_states=module_states,
                    active_alerts=self._process_alerts(all_alerts),
                    performance_summary=self._summarize_performance(all_metrics),
                    resource_usage=self._summarize_resources(all_metrics),
                    recommendations=self._generate_recommendations(module_states, all_alerts)
                )
                
                # 保存到历史记录
                self._state_history.append(copy.deepcopy(self._monitoring_state))
                
                # 保存到数据库
                self._save_to_database()
                
                processing_time = time.time() - start_time
                logger.debug(f"状态聚合完成，耗时: {processing_time:.2f}秒")
                
            except Exception as e:
                logger.error(f"状态聚合异常: {e}")
    
    def _get_monitor_state(self, module_name: str, monitor) -> Tuple[Dict[str, Any], List[MonitoringMetrics], List[Dict[str, Any]]]:
        """获取单个监控器状态（兼容真实和模拟监控器）"""
        try:
            # 获取监控器状态数据
            status_data = {'status': 'UNKNOWN', 'health': 'UNKNOWN', 'metrics': {}, 'alerts': []}
            
            # 方法1: 使用get_status（模拟监控器）
            if hasattr(monitor, 'get_status'):
                try:
                    status_data.update(monitor.get_status())
                except Exception as e:
                    logger.warning(f"模块{module_name} get_status方法调用失败: {e}")
            
            # 方法2: 使用get_current_metrics（真实监控器）
            if hasattr(monitor, 'get_current_metrics') and not status_data.get('metrics'):
                try:
                    metrics_data = monitor.get_current_metrics()
                    if isinstance(metrics_data, dict):
                        status_data['metrics'].update(metrics_data)
                except Exception as e:
                    logger.warning(f"模块{module_name} get_current_metrics方法调用失败: {e}")
            
            # 方法3: 使用get_system_health（真实监控器）
            if hasattr(monitor, 'get_system_health'):
                try:
                    health_status = monitor.get_system_health()
                    status_data['health'] = str(health_status)
                except Exception as e:
                    logger.warning(f"模块{module_name} get_system_health方法调用失败: {e}")
            
            # 方法4: 使用get_active_alerts（真实监控器）
            if hasattr(monitor, 'get_active_alerts'):
                try:
                    alerts_data = monitor.get_active_alerts()
                    if isinstance(alerts_data, list):
                        status_data['alerts'] = alerts_data
                except Exception as e:
                    logger.warning(f"模块{module_name} get_active_alerts方法调用失败: {e}")
            
            # 提取指标数据
            metrics = []
            if 'metrics' in status_data:
                for metric_name, metric_value in status_data['metrics'].items():
                    if isinstance(metric_value, (int, float)):  # 只处理数值型指标
                        metrics.append(MonitoringMetrics(
                            timestamp=datetime.now(),
                            source_module=module_name,
                            metric_name=metric_name,
                            metric_value=metric_value,
                            metric_unit='percent' if 'usage' in metric_name.lower() else 'count',
                            status=status_data.get('status', 'UNKNOWN')
                        ))
            
            # 提取告警数据
            alerts = []
            if 'alerts' in status_data:
                alerts = status_data['alerts']
            
            return status_data, metrics, alerts
            
        except Exception as e:
            logger.error(f"获取{module_name}监控器状态失败: {e}")
            return {'status': 'ERROR', 'health': 'ERROR', 'error': str(e)}, [], []
    
    def _analyze_overall_status(self, module_states: Dict[str, Dict], all_alerts: List[Dict]) -> AggregateStatus:
        """分析整体状态"""
        error_modules = 0
        warning_modules = 0
        
        for module_name, state in module_states.items():
            status = state.get('status', 'UNKNOWN')
            health = state.get('health', 'UNKNOWN')
            
            if status in ['ERROR', 'CRITICAL'] or health in ['ERROR', 'CRITICAL']:
                error_modules += 1
            elif status in ['WARNING'] or health in ['WARNING']:
                warning_modules += 1
        
        # 根据错误模块比例确定整体状态
        total_modules = len(module_states)
        if total_modules == 0:
            return AggregateStatus.UNKNOWN
        
        error_rate = error_modules / total_modules
        warning_rate = warning_modules / total_modules
        
        if error_rate > 0.5 or any(alert.get('severity') == 'CRITICAL' for alert in all_alerts):
            return AggregateStatus.CRITICAL
        elif error_rate > 0.2 or warning_rate > 0.3:
            return AggregateStatus.WARNING
        elif error_rate == 0 and warning_rate < 0.1:
            return AggregateStatus.HEALTHY
        else:
            return AggregateStatus.WARNING
    
    def _create_system_overview(self, module_states: Dict[str, Dict], all_alerts: List[Dict]) -> SystemOverview:
        """创建系统概览"""
        active_modules = sum(1 for state in module_states.values() 
                           if state.get('status') == 'RUNNING')
        
        total_alerts = len(all_alerts)
        critical_alerts = sum(1 for alert in all_alerts 
                            if alert.get('severity') == 'CRITICAL')
        warning_alerts = sum(1 for alert in all_alerts 
                           if alert.get('severity') in ['WARNING', 'MEDIUM'])
        
        # 计算系统健康评分
        health_score = self._calculate_health_score(module_states, all_alerts)
        
        return SystemOverview(
            timestamp=datetime.now(),
            overall_status=self._analyze_overall_status(module_states, all_alerts),
            active_modules=active_modules,
            total_alerts=total_alerts,
            critical_alerts=critical_alerts,
            warning_alerts=warning_alerts,
            system_health_score=health_score,
            module_statuses={name: state.get('status', 'UNKNOWN') 
                           for name, state in module_states.items()}
        )
    
    def _calculate_health_score(self, module_states: Dict[str, Dict], all_alerts: List[Dict]) -> float:
        """计算系统健康评分"""
        if not module_states:
            return 0.0
        
        total_score = 0.0
        
        for module_name, state in module_states.items():
            status = state.get('status', 'UNKNOWN')
            health = state.get('health', 'UNKNOWN')
            
            # 根据状态评分
            if status == 'RUNNING' and health in ['GOOD', 'EXCELLENT']:
                score = 100.0
            elif status == 'RUNNING' and health in ['WARNING']:
                score = 70.0
            elif status in ['MAINTENANCE']:
                score = 50.0
            else:
                score = 20.0  # 错误状态
            
            total_score += score
        
        # 扣分：告警影响
        for alert in all_alerts:
            severity = alert.get('severity', 'LOW')
            if severity == 'CRITICAL':
                total_score -= 10.0
            elif severity == 'HIGH':
                total_score -= 5.0
            elif severity == 'MEDIUM':
                total_score -= 2.0
        
        # 标准化评分
        base_score = total_score / len(module_states)
        return max(0.0, min(100.0, base_score))
    
    def _process_alerts(self, all_alerts: List[Dict]) -> List[AggregateAlert]:
        """处理告警数据"""
        processed_alerts = []
        
        for i, alert in enumerate(all_alerts):
            aggregate_alert = AggregateAlert(
                id=f"agg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                timestamp=datetime.now(),
                source_modules=[alert.get('source', 'unknown')],
                alert_type=alert.get('type', 'GENERAL'),
                severity=AlertSeverity(alert.get('severity', 'LOW').upper()),
                title=alert.get('title', 'Alert'),
                description=alert.get('description', ''),
                metrics=[],
                resolved=alert.get('resolved', False)
            )
            processed_alerts.append(aggregate_alert)
        
        return processed_alerts
    
    def _summarize_performance(self, all_metrics: List[MonitoringMetrics]) -> Dict[str, Any]:
        """汇总性能数据"""
        performance_data = defaultdict(list)
        
        for metric in all_metrics:
            if 'cpu' in metric.metric_name.lower() or 'memory' in metric.metric_name.lower():
                performance_data['system_performance'].append(metric.metric_value)
            elif 'response_time' in metric.metric_name.lower():
                performance_data['response_times'].append(metric.metric_value)
            elif 'throughput' in metric.metric_name.lower():
                performance_data['throughput'].append(metric.metric_value)
        
        summary = {}
        for category, values in performance_data.items():
            if values:
                summary[category] = {
                    'average': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return summary
    
    def _summarize_resources(self, all_metrics: List[MonitoringMetrics]) -> Dict[str, Any]:
        """汇总资源使用数据"""
        resource_data = defaultdict(list)
        
        for metric in all_metrics:
            if 'usage' in metric.metric_name.lower() or 'utilization' in metric.metric_name.lower():
                resource_data['utilization'].append(metric.metric_value)
        
        summary = {}
        if resource_data['utilization']:
            values = resource_data['utilization']
            summary['resource_utilization'] = {
                'average': statistics.mean(values),
                'max': max(values),
                'current': values[-1] if values else 0
            }
        
        return summary
    
    def _generate_recommendations(self, module_states: Dict[str, Dict], all_alerts: List[Dict]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于错误模块的建议
        error_modules = [name for name, state in module_states.items() 
                        if state.get('status') in ['ERROR', 'CRITICAL']]
        if error_modules:
            recommendations.append(f"检查错误模块: {', '.join(error_modules)}")
        
        # 基于告警的建议
        critical_alerts = [alert for alert in all_alerts 
                         if alert.get('severity') == 'CRITICAL']
        if critical_alerts:
            recommendations.append("存在严重告警，需要立即处理")
        
        # 基于资源使用的建议
        for module_name, state in module_states.items():
            metrics = state.get('metrics', {})
            for metric_name, value in metrics.items():
                if 'usage' in metric_name and value > 80:
                    recommendations.append(f"{module_name}的{metric_name}过高({value:.1f}%)，建议优化")
        
        if not recommendations:
            recommendations.append("系统运行正常")
        
        return recommendations
    
    def _cleanup_old_data(self):
        """清理旧数据"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.config.get('retention_days', 30))
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 清理监控状态表
            cursor.execute('DELETE FROM monitoring_state WHERE timestamp < ?', (cutoff_time,))
            
            # 清理告警历史表
            cursor.execute('DELETE FROM alert_history WHERE timestamp < ?', (cutoff_time,))
            
            # 清理指标历史表
            cursor.execute('DELETE FROM metrics_history WHERE timestamp < ?', (cutoff_time,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"清理完成，删除了{deleted_count}条旧记录")
            
        except Exception as e:
            logger.error(f"清理数据失败: {e}")
    
    def _save_to_database(self):
        """保存到数据库"""
        if not self._monitoring_state:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 保存监控状态
            state = self._monitoring_state
            cursor.execute('''
                INSERT INTO monitoring_state 
                (timestamp, overall_status, active_modules, total_alerts, 
                 critical_alerts, warning_alerts, health_score, state_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                state.timestamp,
                state.system_overview.overall_status.value,
                state.system_overview.active_modules,
                state.system_overview.total_alerts,
                state.system_overview.critical_alerts,
                state.system_overview.warning_alerts,
                state.system_overview.system_health_score,
                json.dumps(asdict(state), default=str)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"保存到数据库失败: {e}")
    
    # 公共接口方法
    def get_current_state(self) -> Optional[MonitoringState]:
        """获取当前监控状态"""
        with self._lock:
            return copy.deepcopy(self._monitoring_state) if self._monitoring_state else None
    
    def get_system_overview(self) -> Optional[SystemOverview]:
        """获取系统概览"""
        current_state = self.get_current_state()
        return current_state.system_overview if current_state else None
    
    def get_module_status(self, module_name: str) -> Optional[Dict[str, Any]]:
        """获取指定模块状态"""
        current_state = self.get_current_state()
        return current_state.module_states.get(module_name) if current_state else None
    
    def get_all_alerts(self, include_resolved: bool = False) -> List[AggregateAlert]:
        """获取所有告警"""
        current_state = self.get_current_state()
        if not current_state:
            return []
        
        alerts = current_state.active_alerts
        if not include_resolved:
            alerts = [alert for alert in alerts if not alert.resolved]
        
        return alerts
    
    def get_health_score(self) -> float:
        """获取系统健康评分"""
        overview = self.get_system_overview()
        return overview.system_health_score if overview else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        current_state = self.get_current_state()
        return current_state.performance_summary if current_state else {}
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        current_state = self.get_current_state()
        return current_state.resource_usage if current_state else {}
    
    def get_recommendations(self) -> List[str]:
        """获取系统建议"""
        current_state = self.get_current_state()
        return current_state.recommendations if current_state else []
    
    def export_monitoring_report(self, format_type: str = 'json') -> str:
        """导出监控报告"""
        current_state = self.get_current_state()
        if not current_state:
            return "{}"
        
        if format_type.lower() == 'json':
            return json.dumps(asdict(current_state), default=str, indent=2, ensure_ascii=False)
        elif format_type.lower() == 'summary':
            # 生成文本摘要报告
            return self._generate_text_report(current_state)
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")
    
    def _generate_text_report(self, state: MonitoringState) -> str:
        """生成文本格式报告"""
        overview = state.system_overview
        lines = []
        
        lines.append("=" * 60)
        lines.append("M9 监控状态聚合报告")
        lines.append("=" * 60)
        lines.append(f"生成时间: {state.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"整体状态: {overview.overall_status.value}")
        lines.append(f"系统健康评分: {overview.system_health_score:.1f}/100")
        lines.append(f"活跃模块: {overview.active_modules}/8")
        lines.append(f"告警统计: 总计{overview.total_alerts}个, 严重{overview.critical_alerts}个, 警告{overview.warning_alerts}个")
        lines.append("")
        
        lines.append("模块状态:")
        lines.append("-" * 40)
        for module_name, status in overview.module_statuses.items():
            lines.append(f"  {module_name}: {status}")
        lines.append("")
        
        if state.active_alerts:
            lines.append("活动告警:")
            lines.append("-" * 40)
            for alert in state.active_alerts:
                lines.append(f"  {alert.severity.value}: {alert.title}")
                lines.append(f"    来源: {', '.join(alert.source_modules)}")
                lines.append(f"    描述: {alert.description}")
                lines.append("")
        
        if state.recommendations:
            lines.append("系统建议:")
            lines.append("-" * 40)
            for i, rec in enumerate(state.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_historical_data(self, hours: int = 24) -> List[MonitoringState]:
        """获取历史数据"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        historical_data = [state for state in self._state_history 
                          if state.timestamp >= cutoff_time]
        return list(historical_data)
    
    def search_alerts(self, criteria: Dict[str, Any]) -> List[AggregateAlert]:
        """搜索告警"""
        all_alerts = self.get_all_alerts(include_resolved=True)
        filtered_alerts = []
        
        for alert in all_alerts:
            match = True
            
            # 按严重级别筛选
            if 'severity' in criteria and alert.severity.value != criteria['severity']:
                match = False
            
            # 按来源模块筛选
            if 'source_module' in criteria:
                if not any(criteria['source_module'] in source for source in alert.source_modules):
                    match = False
            
            # 按时间筛选
            if 'time_range' in criteria:
                start_time, end_time = criteria['time_range']
                if not (start_time <= alert.timestamp <= end_time):
                    match = False
            
            if match:
                filtered_alerts.append(alert)
        
        return filtered_alerts
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()


def example_usage():
    """使用示例"""
    # 配置
    config = {
        'aggregation_interval': 30,
        'cleanup_interval': 3600,
        'retention_days': 30,
        'log_level': logging.INFO
    }
    
    # 创建聚合器
    with MonitoringStateAggregator(config) as aggregator:
        # 等待一些数据收集
        time.sleep(5)
        
        # 获取系统概览
        overview = aggregator.get_system_overview()
        print(f"系统状态: {overview.overall_status.value}")
        print(f"健康评分: {overview.system_health_score:.1f}")
        
        # 获取所有告警
        alerts = aggregator.get_all_alerts()
        print(f"活动告警数量: {len(alerts)}")
        
        # 获取模块状态
        for module_name in ['M1', 'M2', 'M3']:
            status = aggregator.get_module_status(module_name)
            if status:
                print(f"{module_name}: {status.get('status', 'UNKNOWN')}")
        
        # 获取性能摘要
        performance = aggregator.get_performance_summary()
        print(f"性能摘要: {performance}")
        
        # 导出报告
        report = aggregator.export_monitoring_report('summary')
        print("\n监控报告:")
        print(report)


if __name__ == "__main__":
    example_usage()