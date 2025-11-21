"""
S9服务状态聚合器

该模块提供了完整的服务状态聚合和监控功能，包括状态收集、数据聚合、
状态分析、报告生成、实时监控、预警机制、历史记录和可视化仪表板。
"""

import json
import time
import threading
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics

# 尝试导入matplotlib，如果失败则设置为None
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    mdates = None
    MATPLOTLIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("matplotlib未安装，可视化功能将不可用")


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """服务状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """预警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ServiceInfo:
    """服务信息数据类"""
    service_id: str
    name: str
    endpoint: str
    status: ServiceStatus
    last_check: datetime
    response_time: float
    error_count: int
    metadata: Dict[str, Any]


@dataclass
class AlertInfo:
    """预警信息数据类"""
    alert_id: str
    service_id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    resolved: bool = False


@dataclass
class StatusReport:
    """状态报告数据类"""
    report_id: str
    timestamp: datetime
    total_services: int
    healthy_services: int
    warning_services: int
    critical_services: int
    offline_services: int
    avg_response_time: float
    alerts: List[AlertInfo]
    summary: str


class StatusCollector:
    """状态收集器"""
    
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.collectors: Dict[str, Callable] = {}
        
    def register_service(self, service_id: str, name: str, endpoint: str, 
                        collector_func: Callable, metadata: Dict[str, Any] = None):
        """注册服务"""
        self.services[service_id] = ServiceInfo(
            service_id=service_id,
            name=name,
            endpoint=endpoint,
            status=ServiceStatus.UNKNOWN,
            last_check=datetime.now(),
            response_time=0.0,
            error_count=0,
            metadata=metadata or {}
        )
        self.collectors[service_id] = collector_func
        logger.info(f"注册服务: {service_id} - {name}")
    
    def collect_status(self, service_id: str) -> ServiceInfo:
        """收集指定服务状态"""
        if service_id not in self.collectors:
            raise ValueError(f"未找到服务收集器: {service_id}")
        
        try:
            collector_func = self.collectors[service_id]
            result = collector_func()
            
            service_info = self.services[service_id]
            service_info.status = result.get('status', ServiceStatus.UNKNOWN)
            service_info.response_time = result.get('response_time', 0.0)
            service_info.last_check = datetime.now()
            
            if service_info.status == ServiceStatus.CRITICAL:
                service_info.error_count += 1
            else:
                service_info.error_count = 0
                
            return service_info
            
        except Exception as e:
            logger.error(f"收集服务 {service_id} 状态失败: {e}")
            service_info = self.services[service_id]
            service_info.status = ServiceStatus.OFFLINE
            service_info.last_check = datetime.now()
            return service_info
    
    def collect_all_status(self) -> Dict[str, ServiceInfo]:
        """收集所有服务状态"""
        results = {}
        for service_id in self.services:
            results[service_id] = self.collect_status(service_id)
        return results


class DataAggregator:
    """数据聚合器"""
    
    def __init__(self):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def aggregate_service_metrics(self, services: Dict[str, ServiceInfo]) -> Dict[str, Any]:
        """聚合服务指标"""
        if not services:
            return {}
        
        total_response_time = sum(s.response_time for s in services.values() if s.response_time > 0)
        avg_response_time = total_response_time / len(services) if services else 0
        
        status_counts = defaultdict(int)
        for service in services.values():
            status_counts[service.status.value] += 1
        
        # 记录历史指标
        timestamp = datetime.now()
        self.metrics_history['response_time'].append((timestamp, avg_response_time))
        self.metrics_history['total_services'].append((timestamp, len(services)))
        
        for status, count in status_counts.items():
            self.metrics_history[f'{status}_count'].append((timestamp, count))
        
        return {
            'total_services': len(services),
            'avg_response_time': avg_response_time,
            'status_distribution': dict(status_counts),
            'timestamp': timestamp
        }
    
    def get_trend_analysis(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """获取趋势分析"""
        if metric_name not in self.metrics_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [(ts, value) for ts, value in self.metrics_history[metric_name] 
                      if ts >= cutoff_time]
        
        if len(recent_data) < 2:
            return {'trend': 'insufficient_data', 'change_rate': 0}
        
        values = [value for _, value in recent_data]
        first_value = values[0]
        last_value = values[-1]
        
        if first_value == 0:
            change_rate = 0
        else:
            change_rate = ((last_value - first_value) / first_value) * 100
        
        if change_rate > 5:
            trend = 'increasing'
        elif change_rate < -5:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change_rate': change_rate,
            'min_value': min(values),
            'max_value': max(values),
            'avg_value': statistics.mean(values),
            'data_points': len(recent_data)
        }


class StatusAnalyzer:
    """状态分析器"""
    
    def __init__(self):
        self.thresholds = {
            'response_time_warning': 2.0,  # 秒
            'response_time_critical': 5.0,
            'error_count_warning': 3,
            'error_count_critical': 5
        }
    
    def analyze_service_health(self, service: ServiceInfo) -> ServiceStatus:
        """分析服务健康状态"""
        # 基于响应时间判断
        if service.response_time > self.thresholds['response_time_critical']:
            return ServiceStatus.CRITICAL
        elif service.response_time > self.thresholds['response_time_warning']:
            return ServiceStatus.WARNING
        
        # 基于错误次数判断
        if service.error_count >= self.thresholds['error_count_critical']:
            return ServiceStatus.CRITICAL
        elif service.error_count >= self.thresholds['error_count_warning']:
            return ServiceStatus.WARNING
        
        # 基于状态判断
        if service.status == ServiceStatus.OFFLINE:
            return ServiceStatus.OFFLINE
        
        return ServiceStatus.HEALTHY
    
    def detect_anomalies(self, services: Dict[str, ServiceInfo]) -> List[Dict[str, Any]]:
        """检测异常"""
        anomalies = []
        
        for service in services.values():
            # 检测响应时间异常
            if service.response_time > self.thresholds['response_time_critical']:
                anomalies.append({
                    'type': 'high_response_time',
                    'service_id': service.service_id,
                    'value': service.response_time,
                    'threshold': self.thresholds['response_time_critical'],
                    'severity': 'critical'
                })
            
            # 检测错误率异常
            if service.error_count >= self.thresholds['error_count_critical']:
                anomalies.append({
                    'type': 'high_error_rate',
                    'service_id': service.service_id,
                    'value': service.error_count,
                    'threshold': self.thresholds['error_count_critical'],
                    'severity': 'critical'
                })
        
        return anomalies


class AlertSystem:
    """预警系统"""
    
    def __init__(self):
        self.alerts: List[AlertInfo] = []
        self.alert_callbacks: List[Callable] = []
        self.alert_rules: Dict[str, Callable] = {}
        
    def add_alert_rule(self, rule_name: str, rule_func: Callable):
        """添加预警规则"""
        self.alert_rules[rule_name] = rule_func
        logger.info(f"添加预警规则: {rule_name}")
    
    def trigger_alert(self, service_id: str, level: AlertLevel, message: str):
        """触发预警"""
        alert_id = f"alert_{int(time.time())}_{service_id}"
        alert = AlertInfo(
            alert_id=alert_id,
            service_id=service_id,
            level=level,
            message=message,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        logger.warning(f"触发预警 [{level.value}]: {service_id} - {message}")
        
        # 调用预警回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"预警回调执行失败: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """添加预警回调"""
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[AlertInfo]:
        """获取活跃预警"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """解决预警"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"解决预警: {alert_id}")
                break


class HistoryManager:
    """历史记录管理器"""
    
    def __init__(self, db_path: str = "service_status_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS service_status_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service_id TEXT NOT NULL,
                service_name TEXT NOT NULL,
                status TEXT NOT NULL,
                response_time REAL,
                error_count INTEGER,
                timestamp DATETIME NOT NULL,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT NOT NULL,
                service_id TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_service_status(self, service: ServiceInfo):
        """保存服务状态到历史记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO service_status_history 
            (service_id, service_name, status, response_time, error_count, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            service.service_id,
            service.name,
            service.status.value,
            service.response_time,
            service.error_count,
            service.last_check,
            json.dumps(service.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def save_alert(self, alert: AlertInfo):
        """保存预警到历史记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alert_history 
            (alert_id, service_id, level, message, timestamp, resolved)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            alert.alert_id,
            alert.service_id,
            alert.level.value,
            alert.message,
            alert.timestamp,
            alert.resolved
        ))
        
        conn.commit()
        conn.close()
    
    def get_service_history(self, service_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """获取服务历史记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cursor.execute('''
            SELECT service_id, service_name, status, response_time, error_count, timestamp, metadata
            FROM service_status_history
            WHERE service_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
        ''', (service_id, cutoff_time))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'service_id': row[0],
                'service_name': row[1],
                'status': row[2],
                'response_time': row[3],
                'error_count': row[4],
                'timestamp': row[5],
                'metadata': json.loads(row[6]) if row[6] else {}
            })
        
        conn.close()
        return results


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, history_manager: HistoryManager):
        self.history_manager = history_manager
    
    def generate_status_report(self, services: Dict[str, ServiceInfo], 
                             alerts: List[AlertInfo]) -> StatusReport:
        """生成状态报告"""
        total_services = len(services)
        healthy_services = sum(1 for s in services.values() if s.status == ServiceStatus.HEALTHY)
        warning_services = sum(1 for s in services.values() if s.status == ServiceStatus.WARNING)
        critical_services = sum(1 for s in services.values() if s.status == ServiceStatus.CRITICAL)
        offline_services = sum(1 for s in services.values() if s.status == ServiceStatus.OFFLINE)
        
        response_times = [s.response_time for s in services.values() if s.response_time > 0]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        # 生成摘要
        summary = self._generate_summary(total_services, healthy_services, warning_services, 
                                       critical_services, offline_services, avg_response_time)
        
        report = StatusReport(
            report_id=f"report_{int(time.time())}",
            timestamp=datetime.now(),
            total_services=total_services,
            healthy_services=healthy_services,
            warning_services=warning_services,
            critical_services=critical_services,
            offline_services=offline_services,
            avg_response_time=avg_response_time,
            alerts=alerts,
            summary=summary
        )
        
        return report
    
    def _generate_summary(self, total: int, healthy: int, warning: int, 
                         critical: int, offline: int, avg_response_time: float) -> str:
        """生成摘要"""
        health_rate = (healthy / total * 100) if total > 0 else 0
        
        summary = f"系统总体健康度: {health_rate:.1f}%\\n"
        summary += f"总服务数: {total}\\n"
        summary += f"健康服务: {healthy}\\n"
        summary += f"警告服务: {warning}\\n"
        summary += f"严重服务: {critical}\\n"
        summary += f"离线服务: {offline}\\n"
        summary += f"平均响应时间: {avg_response_time:.2f}秒"
        
        return summary
    
    def export_report_json(self, report: StatusReport) -> str:
        """导出JSON格式报告"""
        report_dict = asdict(report)
        # 转换datetime对象为字符串
        report_dict['timestamp'] = report.timestamp.isoformat()
        for alert in report_dict['alerts']:
            alert['timestamp'] = alert['timestamp'].isoformat()
            # 转换枚举为字符串
            alert['level'] = alert['level'].value if hasattr(alert['level'], 'value') else str(alert['level'])
        
        return json.dumps(report_dict, indent=2, ensure_ascii=False, default=str)
    
    def export_report_html(self, report: StatusReport) -> str:
        """导出HTML格式报告"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>服务状态报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .alerts {{ background-color: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>服务状态报告</h1>
                <p>生成时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>系统摘要</h2>
                <pre>{report.summary}</pre>
            </div>
            
            <div class="metrics">
                <h2>关键指标</h2>
                <div class="metric">总服务数: {report.total_services}</div>
                <div class="metric">健康服务: {report.healthy_services}</div>
                <div class="metric">警告服务: {report.warning_services}</div>
                <div class="metric">严重服务: {report.critical_services}</div>
                <div class="metric">离线服务: {report.offline_services}</div>
                <div class="metric">平均响应时间: {report.avg_response_time:.2f}秒</div>
            </div>
            
            <div class="alerts">
                <h2>活跃预警</h2>
                {len(report.alerts)}
            </div>
        </body>
        </html>
        """
        return html


class StatusMonitor:
    """状态监控器"""
    
    def __init__(self, status_collector: StatusCollector, alert_system: AlertSystem,
                 history_manager: HistoryManager, check_interval: int = 30):
        self.status_collector = status_collector
        self.alert_system = alert_system
        self.history_manager = history_manager
        self.check_interval = check_interval
        self.is_monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            logger.warning("监控已在运行中")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("开始服务状态监控")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("停止服务状态监控")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集所有服务状态
                services = self.status_collector.collect_all_status()
                
                # 保存历史记录
                for service in services.values():
                    self.history_manager.save_service_status(service)
                
                # 检查预警条件
                for service in services.values():
                    if service.status == ServiceStatus.CRITICAL:
                        self.alert_system.trigger_alert(
                            service.service_id,
                            AlertLevel.CRITICAL,
                            f"服务 {service.name} 处于严重状态"
                        )
                    elif service.status == ServiceStatus.WARNING:
                        self.alert_system.trigger_alert(
                            service.service_id,
                            AlertLevel.WARNING,
                            f"服务 {service.name} 处于警告状态"
                        )
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(5)  # 异常时短暂等待


class Dashboard:
    """可视化仪表板"""
    
    def __init__(self, data_aggregator: DataAggregator):
        self.data_aggregator = data_aggregator
        
    def generate_status_chart(self, services: Dict[str, ServiceInfo], 
                            output_path: str = "service_status_chart.png"):
        """生成服务状态图表"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib未安装，无法生成图表")
            return
        
        if not services:
            logger.warning("没有服务数据可绘制")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 服务状态分布饼图
        status_counts = defaultdict(int)
        for service in services.values():
            status_counts[service.status.value] += 1
        
        labels = list(status_counts.keys())
        sizes = list(status_counts.values())
        colors = ['#4CAF50', '#FF9800', '#F44336', '#9E9E9E', '#2196F3']
        
        ax1.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%')
        ax1.set_title('服务状态分布')
        
        # 响应时间柱状图
        service_names = [service.name for service in services.values()]
        response_times = [service.response_time for service in services.values()]
        
        bars = ax2.bar(service_names, response_times, color='skyblue')
        ax2.set_title('服务响应时间')
        ax2.set_ylabel('响应时间 (秒)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加响应时间数值标签
        for bar, time_val in zip(bars, response_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"生成状态图表: {output_path}")
    
    def generate_trend_chart(self, metric_name: str, hours: int = 24,
                           output_path: str = "trend_chart.png"):
        """生成趋势图表"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib未安装，无法生成图表")
            return
        
        trend_data = self.data_aggregator.get_trend_analysis(metric_name, hours)
        
        if not trend_data or trend_data.get('data_points', 0) < 2:
            logger.warning(f"趋势数据不足: {metric_name}")
            return
        
        # 获取历史数据
        history_data = self.data_aggregator.metrics_history[metric_name]
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_data = [(ts, value) for ts, value in history_data if ts >= cutoff_time]
        
        if not filtered_data:
            logger.warning(f"无历史数据: {metric_name}")
            return
        
        timestamps, values = zip(*filtered_data)
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, values, marker='o', linewidth=2, markersize=4)
        plt.title(f'{metric_name} 趋势图 (过去{hours}小时)')
        plt.xlabel('时间')
        plt.ylabel(metric_name)
        plt.grid(True, alpha=0.3)
        
        # 格式化时间轴
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"生成趋势图表: {output_path}")


class ServiceStatusAggregator:
    """服务状态聚合器主类"""
    
    def __init__(self, db_path: str = "service_status_history.db"):
        self.status_collector = StatusCollector()
        self.data_aggregator = DataAggregator()
        self.status_analyzer = StatusAnalyzer()
        self.alert_system = AlertSystem()
        self.history_manager = HistoryManager(db_path)
        self.report_generator = ReportGenerator(self.history_manager)
        self.status_monitor = StatusMonitor(
            self.status_collector, 
            self.alert_system, 
            self.history_manager
        )
        self.dashboard = Dashboard(self.data_aggregator)
        
        # 设置默认预警回调
        self.alert_system.add_alert_callback(self._default_alert_callback)
    
    def _default_alert_callback(self, alert: AlertInfo):
        """默认预警回调"""
        self.history_manager.save_alert(alert)
        print(f"🚨 预警 [{alert.level.value.upper()}]: {alert.service_id} - {alert.message}")
    
    def register_service(self, service_id: str, name: str, endpoint: str,
                        collector_func: Callable, metadata: Dict[str, Any] = None):
        """注册服务"""
        self.status_collector.register_service(service_id, name, endpoint, 
                                             collector_func, metadata)
    
    def collect_status(self, service_id: str = None) -> Dict[str, ServiceInfo]:
        """收集服务状态"""
        if service_id:
            return {service_id: self.status_collector.collect_status(service_id)}
        else:
            return self.status_collector.collect_all_status()
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """获取聚合指标"""
        services = self.status_collector.services
        return self.data_aggregator.aggregate_service_metrics(services)
    
    def analyze_status(self) -> List[Dict[str, Any]]:
        """分析服务状态"""
        services = self.status_collector.services
        return self.status_analyzer.detect_anomalies(services)
    
    def generate_report(self) -> StatusReport:
        """生成状态报告"""
        services = self.status_collector.services
        active_alerts = self.alert_system.get_active_alerts()
        return self.report_generator.generate_status_report(services, active_alerts)
    
    def start_monitoring(self, interval: int = 30):
        """开始监控"""
        self.status_monitor.check_interval = interval
        self.status_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """停止监控"""
        self.status_monitor.stop_monitoring()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        services = self.status_collector.services
        metrics = self.get_aggregated_metrics()
        alerts = self.alert_system.get_active_alerts()
        
        return {
            'services': {sid: asdict(service) for sid, service in services.items()},
            'metrics': metrics,
            'alerts': [asdict(alert) for alert in alerts],
            'timestamp': datetime.now().isoformat()
        }
    
    def export_dashboard_json(self, file_path: str = "dashboard_data.json"):
        """导出仪表板数据为JSON"""
        data = self.get_dashboard_data()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"导出仪表板数据: {file_path}")
    
    def generate_charts(self, output_dir: str = "./charts"):
        """生成图表"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        services = self.status_collector.services
        
        # 生成状态图表
        status_chart_path = os.path.join(output_dir, "service_status_chart.png")
        self.dashboard.generate_status_chart(services, status_chart_path)
        
        # 生成趋势图表
        trend_chart_path = os.path.join(output_dir, "response_time_trend.png")
        self.dashboard.generate_trend_chart('response_time', 24, trend_chart_path)
        
        return {
            'status_chart': status_chart_path,
            'trend_chart': trend_chart_path
        }


# 示例收集器函数
def example_service_collector():
    """示例服务收集器函数"""
    import random
    import requests
    
    # 模拟检查结果
    status_options = [ServiceStatus.HEALTHY, ServiceStatus.WARNING, ServiceStatus.CRITICAL]
    status = random.choice(status_options)
    response_time = random.uniform(0.1, 3.0)
    
    return {
        'status': status,
        'response_time': response_time,
        'timestamp': datetime.now()
    }


if __name__ == "__main__":
    # 示例使用
    aggregator = ServiceStatusAggregator()
    
    # 注册示例服务
    aggregator.register_service(
        service_id="web_service",
        name="Web服务",
        endpoint="http://localhost:8080",
        collector_func=example_service_collector
    )
    
    aggregator.register_service(
        service_id="api_service", 
        name="API服务",
        endpoint="http://localhost:8081",
        collector_func=example_service_collector
    )
    
    # 收集状态
    services = aggregator.collect_status()
    print("服务状态:")
    for service_id, service in services.items():
        print(f"  {service_id}: {service.status.value} ({service.response_time:.2f}s)")
    
    # 生成报告
    report = aggregator.generate_report()
    print(f"\\n状态报告摘要:")
    print(report.summary)
    
    # 生成图表
    if MATPLOTLIB_AVAILABLE:
        charts = aggregator.generate_charts()
        print(f"\\n生成的图表:")
        for chart_type, path in charts.items():
            print(f"  {chart_type}: {path}")
    else:
        print("\\nmatplotlib未安装，跳过图表生成")