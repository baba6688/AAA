"""
S7监控服务主实现文件

提供完整的系统监控、性能监控、告警管理等功能
"""

import psutil
import time
import json
import threading
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import os
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricData:
    """监控指标数据结构"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric: str
    operator: str  # >, <, >=, <=, ==
    threshold: float
    level: AlertLevel
    enabled: bool = True
    description: str = ""


@dataclass
class Alert:
    """告警信息"""
    id: str
    rule_name: str
    metric: str
    value: float
    threshold: float
    level: AlertLevel
    message: str
    timestamp: datetime
    resolved: bool = False


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        return psutil.cpu_percent(interval=1)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024**3),  # GB
            'available': memory.available / (1024**3),  # GB
            'used': memory.used / (1024**3),  # GB
            'percent': memory.percent
        }
    
    def get_disk_usage(self, path: str = '/') -> Dict[str, float]:
        """获取磁盘使用情况"""
        disk = psutil.disk_usage(path)
        return {
            'total': disk.total / (1024**3),  # GB
            'used': disk.used / (1024**3),  # GB
            'free': disk.free / (1024**3),  # GB
            'percent': (disk.used / disk.total) * 100
        }
    
    def get_network_stats(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'errin': net_io.errin,
            'errout': net_io.errout,
            'dropin': net_io.dropin,
            'dropout': net_io.dropout
        }
    
    def get_process_count(self) -> int:
        """获取进程数量"""
        return len(psutil.pids())
    
    def get_system_load(self) -> List[float]:
        """获取系统负载"""
        try:
            load_avg = os.getloadavg()
            return list(load_avg)
        except AttributeError:
            # Windows系统不支持getloadavg
            return [0.0, 0.0, 0.0]


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
    
    def get_uptime(self) -> float:
        """获取系统运行时间"""
        return time.time() - self.start_time
    
    def get_disk_io_stats(self) -> Dict[str, Any]:
        """获取磁盘I/O统计"""
        disk_io = psutil.disk_io_counters()
        if disk_io:
            return {
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count,
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_time': disk_io.read_time,
                'write_time': disk_io.write_time
            }
        return {}
    
    def get_network_io_stats(self) -> Dict[str, Any]:
        """获取网络I/O统计"""
        net_io = psutil.net_io_counters()
        if net_io:
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        return {}
    
    def get_top_processes(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取CPU使用率最高的进程"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                proc_info = proc.info
                if proc_info['cpu_percent'] > 0:
                    processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # 按CPU使用率排序
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        return processes[:limit]


class AlertManager:
    """告警管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.rules: List[AlertRule] = []
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        self.alert_id_counter = 0
    
    def add_rule(self, rule: AlertRule) -> None:
        """添加告警规则"""
        self.rules.append(rule)
        self.logger.info(f"添加告警规则: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """删除告警规则"""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                self.logger.info(f"删除告警规则: {rule_name}")
                return True
        return False
    
    def check_alerts(self, metrics: Dict[str, MetricData]) -> List[Alert]:
        """检查告警"""
        triggered_alerts = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            metric = metrics.get(rule.metric)
            if not metric:
                continue
            
            # 检查是否触发告警条件
            triggered = False
            if rule.operator == '>' and metric.value > rule.threshold:
                triggered = True
            elif rule.operator == '<' and metric.value < rule.threshold:
                triggered = True
            elif rule.operator == '>=' and metric.value >= rule.threshold:
                triggered = True
            elif rule.operator == '<=' and metric.value <= rule.threshold:
                triggered = True
            elif rule.operator == '==' and metric.value == rule.threshold:
                triggered = True
            
            if triggered:
                alert = self._create_alert(rule, metric)
                triggered_alerts.append(alert)
                self.alerts.append(alert)
                self._handle_alert(alert)
        
        return triggered_alerts
    
    def _create_alert(self, rule: AlertRule, metric: MetricData) -> Alert:
        """创建告警"""
        self.alert_id_counter += 1
        alert_id = f"alert_{self.alert_id_counter}"
        
        message = f"{rule.description}: {metric.name} = {metric.value} {metric.unit}, 阈值: {rule.threshold}"
        
        return Alert(
            id=alert_id,
            rule_name=rule.name,
            metric=rule.metric,
            value=metric.value,
            threshold=rule.threshold,
            level=rule.level,
            message=message,
            timestamp=datetime.now()
        )
    
    def _handle_alert(self, alert: Alert) -> None:
        """处理告警"""
        self.logger.warning(f"告警触发: {alert.message}")
        
        # 调用告警回调函数
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"告警回调执行失败: {e}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                self.logger.info(f"告警已解决: {alert_id}")
                return True
        return False


class MonitoringData:
    """监控数据管理"""
    
    def __init__(self, db_path: str = "monitoring_data.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建指标数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                category TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        
        # 创建告警表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                rule_name TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                threshold REAL NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_metric(self, metric: MetricData) -> None:
        """保存监控指标"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO metrics (name, value, unit, category, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (metric.name, metric.value, metric.unit, metric.category, metric.timestamp.isoformat()))
        
        conn.commit()
        conn.close()
    
    def save_alert(self, alert: Alert) -> None:
        """保存告警"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO alerts 
            (id, rule_name, metric, value, threshold, level, message, timestamp, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.id, alert.rule_name, alert.metric, alert.value,
            alert.threshold, alert.level.value, alert.message,
            alert.timestamp.isoformat(), alert.resolved
        ))
        
        conn.commit()
        conn.close()
    
    def get_metrics(self, category: Optional[str] = None, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """获取监控指标"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM metrics WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [description[0] for description in cursor.description]
        result = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return result
    
    def get_alerts(self, resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """获取告警"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM alerts"
        params = []
        
        if resolved is not None:
            query += " WHERE resolved = ?"
            params.append(resolved)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [description[0] for description in cursor.description]
        result = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return result


class Dashboard:
    """监控仪表板"""
    
    def __init__(self, data_manager: MonitoringData):
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
    
    def generate_system_overview(self) -> Dict[str, Any]:
        """生成系统概览"""
        # 获取最近的监控数据
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        metrics = self.data_manager.get_metrics(start_time=start_time, end_time=end_time)
        
        # 按类别分组
        overview = {}
        for metric in metrics:
            category = metric['category']
            if category not in overview:
                overview[category] = []
            overview[category].append(metric)
        
        return {
            'timestamp': end_time.isoformat(),
            'categories': overview,
            'total_metrics': len(metrics)
        }
    
    def generate_performance_chart(self, metric_name: str, hours: int = 24) -> str:
        """生成性能图表（Base64编码的图像）"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = self.data_manager.get_metrics(
            start_time=start_time, 
            end_time=end_time
        )
        
        # 过滤指定指标
        filtered_metrics = [m for m in metrics if m['name'] == metric_name]
        
        if not filtered_metrics:
            return ""
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in filtered_metrics]
        values = [m['value'] for m in filtered_metrics]
        
        ax.plot(timestamps, values, marker='o', linewidth=2)
        ax.set_title(f'{metric_name} 趋势图')
        ax.set_xlabel('时间')
        ax.set_ylabel('值')
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 转换为Base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def generate_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        overview = self.generate_system_overview()
        
        # 获取告警统计
        all_alerts = self.data_manager.get_alerts()
        active_alerts = self.data_manager.get_alerts(resolved=False)
        
        # 按级别统计告警
        alert_stats = {}
        for alert in all_alerts:
            level = alert['level']
            if level not in alert_stats:
                alert_stats[level] = {'total': 0, 'active': 0}
            alert_stats[level]['total'] += 1
            if not alert['resolved']:
                alert_stats[level]['active'] += 1
        
        return {
            'report_time': datetime.now().isoformat(),
            'system_overview': overview,
            'alert_statistics': alert_stats,
            'total_alerts': len(all_alerts),
            'active_alerts': len(active_alerts),
            'system_status': 'healthy' if len(active_alerts) == 0 else 'warning'
        }


class MonitoringService:
    """S7监控服务主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # 初始化各个组件
        self.system_monitor = SystemMonitor()
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager(self.config.get('alert', {}))
        self.data_manager = MonitoringData(self.config.get('db_path', 'monitoring_data.db'))
        self.dashboard = Dashboard(self.data_manager)
        
        # 运行状态
        self.running = False
        self.monitoring_thread = None
        self.monitoring_interval = self.config.get('monitoring_interval', 60)  # 秒
        
        # 设置默认告警规则
        self._setup_default_alert_rules()
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('S7MonitoringService')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_default_alert_rules(self) -> None:
        """设置默认告警规则"""
        default_rules = [
            AlertRule(
                name="高CPU使用率",
                metric="cpu_usage",
                operator=">",
                threshold=80.0,
                level=AlertLevel.WARNING,
                description="CPU使用率超过80%"
            ),
            AlertRule(
                name="高内存使用率",
                metric="memory_usage",
                operator=">",
                threshold=85.0,
                level=AlertLevel.WARNING,
                description="内存使用率超过85%"
            ),
            AlertRule(
                name="磁盘空间不足",
                metric="disk_usage",
                operator=">",
                threshold=90.0,
                level=AlertLevel.CRITICAL,
                description="磁盘使用率超过90%"
            ),
            AlertRule(
                name="系统负载过高",
                metric="system_load_1min",
                operator=">",
                threshold=2.0,
                level=AlertLevel.WARNING,
                description="系统1分钟负载超过2.0"
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_rule(rule)
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self.running:
            self.logger.warning("监控服务已经在运行")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("监控服务已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("监控服务已停止")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.running:
            try:
                self._collect_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(5)  # 错误时短暂暂停
    
    def _collect_metrics(self) -> None:
        """收集监控指标"""
        current_time = datetime.now()
        
        # 系统监控指标
        cpu_usage = self.system_monitor.get_cpu_usage()
        memory_info = self.system_monitor.get_memory_usage()
        disk_info = self.system_monitor.get_disk_usage()
        network_stats = self.system_monitor.get_network_stats()
        process_count = self.system_monitor.get_process_count()
        system_load = self.system_monitor.get_system_load()
        
        # 性能监控指标
        uptime = self.performance_monitor.get_uptime()
        disk_io = self.performance_monitor.get_disk_io_stats()
        network_io = self.performance_monitor.get_network_io_stats()
        top_processes = self.performance_monitor.get_top_processes()
        
        # 创建指标数据
        metrics = {
            'cpu_usage': MetricData('CPU使用率', cpu_usage, '%', current_time, 'system'),
            'memory_usage': MetricData('内存使用率', memory_info['percent'], '%', current_time, 'system'),
            'memory_total': MetricData('总内存', memory_info['total'], 'GB', current_time, 'system'),
            'memory_used': MetricData('已用内存', memory_info['used'], 'GB', current_time, 'system'),
            'disk_usage': MetricData('磁盘使用率', disk_info['percent'], '%', current_time, 'system'),
            'disk_total': MetricData('总磁盘空间', disk_info['total'], 'GB', current_time, 'system'),
            'disk_used': MetricData('已用磁盘空间', disk_info['used'], 'GB', current_time, 'system'),
            'network_bytes_sent': MetricData('网络发送字节', network_stats['bytes_sent'], 'bytes', current_time, 'network'),
            'network_bytes_recv': MetricData('网络接收字节', network_stats['bytes_recv'], 'bytes', current_time, 'network'),
            'process_count': MetricData('进程数量', process_count, 'count', current_time, 'system'),
            'system_load_1min': MetricData('系统负载(1分钟)', system_load[0], 'load', current_time, 'system'),
            'system_load_5min': MetricData('系统负载(5分钟)', system_load[1], 'load', current_time, 'system'),
            'system_load_15min': MetricData('系统负载(15分钟)', system_load[2], 'load', current_time, 'system'),
            'uptime': MetricData('运行时间', uptime, 'seconds', current_time, 'performance')
        }
        
        # 保存指标数据
        for metric in metrics.values():
            self.data_manager.save_metric(metric)
        
        # 检查告警
        alerts = self.alert_manager.check_alerts(metrics)
        
        # 保存告警
        for alert in alerts:
            self.data_manager.save_alert(alert)
        
        self.logger.debug(f"收集了 {len(metrics)} 个指标，触发了 {len(alerts)} 个告警")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        # 实时获取当前系统状态
        cpu_usage = self.system_monitor.get_cpu_usage()
        memory_info = self.system_monitor.get_memory_usage()
        disk_info = self.system_monitor.get_disk_usage()
        
        # 获取活跃告警
        active_alerts = self.alert_manager.get_active_alerts()
        
        # 判断系统健康状态
        if any(alert.level == AlertLevel.CRITICAL for alert in active_alerts):
            health_status = "critical"
        elif any(alert.level == AlertLevel.ERROR for alert in active_alerts):
            health_status = "error"
        elif any(alert.level == AlertLevel.WARNING for alert in active_alerts):
            health_status = "warning"
        else:
            health_status = "healthy"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health_status': health_status,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_info['percent'],
            'disk_usage': disk_info['percent'],
            'active_alerts': len(active_alerts),
            'alerts': [asdict(alert) for alert in active_alerts]
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        return self.dashboard.generate_system_overview()
    
    def generate_performance_chart(self, metric_name: str, hours: int = 24) -> str:
        """生成性能图表"""
        return self.dashboard.generate_performance_chart(metric_name, hours)
    
    def generate_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        return self.dashboard.generate_report()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查系统资源
            cpu_usage = self.system_monitor.get_cpu_usage()
            memory_info = self.system_monitor.get_memory_usage()
            disk_info = self.system_monitor.get_disk_usage()
            
            # 检查监控服务状态
            monitoring_status = "running" if self.running else "stopped"
            
            # 获取活跃告警
            active_alerts = self.alert_manager.get_active_alerts()
            
            health_status = "healthy"
            issues = []
            
            if cpu_usage > 90:
                health_status = "critical"
                issues.append(f"CPU使用率过高: {cpu_usage}%")
            
            if memory_info['percent'] > 95:
                health_status = "critical"
                issues.append(f"内存使用率过高: {memory_info['percent']}%")
            
            if disk_info['percent'] > 95:
                health_status = "critical"
                issues.append(f"磁盘使用率过高: {disk_info['percent']}%")
            
            if any(alert.level == AlertLevel.CRITICAL for alert in active_alerts):
                health_status = "critical"
                issues.append("存在严重告警")
            elif any(alert.level == AlertLevel.ERROR for alert in active_alerts):
                health_status = "error"
                issues.append("存在错误级别告警")
            elif any(alert.level == AlertLevel.WARNING for alert in active_alerts):
                health_status = "warning"
                issues.append("存在警告级别告警")
            
            return {
                'status': health_status,
                'timestamp': datetime.now().isoformat(),
                'monitoring_service': monitoring_status,
                'system_metrics': {
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_info['percent'],
                    'disk_usage': disk_info['percent']
                },
                'active_alerts': len(active_alerts),
                'issues': issues
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def detect_faults(self) -> List[Dict[str, Any]]:
        """故障检测"""
        faults = []
        
        try:
            # 检测系统资源故障
            cpu_usage = self.system_monitor.get_cpu_usage()
            memory_info = self.system_monitor.get_memory_usage()
            disk_info = self.system_monitor.get_disk_usage()
            
            if cpu_usage > 95:
                faults.append({
                    'type': 'resource',
                    'severity': 'critical',
                    'description': f'CPU使用率严重过高: {cpu_usage}%',
                    'timestamp': datetime.now().isoformat()
                })
            
            if memory_info['percent'] > 98:
                faults.append({
                    'type': 'resource',
                    'severity': 'critical',
                    'description': f'内存使用率严重过高: {memory_info["percent"]}%',
                    'timestamp': datetime.now().isoformat()
                })
            
            if disk_info['percent'] > 98:
                faults.append({
                    'type': 'resource',
                    'severity': 'critical',
                    'description': f'磁盘空间严重不足: {disk_info["percent"]}%',
                    'timestamp': datetime.now().isoformat()
                })
            
            # 检测网络连接故障
            try:
                # 尝试ping本地主机
                result = subprocess.run(['ping', '-c', '1', '127.0.0.1'], 
                                      capture_output=True, timeout=5)
                if result.returncode != 0:
                    faults.append({
                        'type': 'network',
                        'severity': 'error',
                        'description': '本地网络连接异常',
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception:
                faults.append({
                    'type': 'network',
                    'severity': 'error',
                    'description': '网络检测失败',
                    'timestamp': datetime.now().isoformat()
                })
            
            # 检测进程异常
            process_count = self.system_monitor.get_process_count()
            if process_count < 5:  # 假设正常系统至少应有5个进程
                faults.append({
                    'type': 'process',
                    'severity': 'warning',
                    'description': f'进程数量异常少: {process_count}',
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            faults.append({
                'type': 'monitoring',
                'severity': 'error',
                'description': f'故障检测过程出错: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })
        
        return faults
    
    def add_custom_metric(self, name: str, value: float, unit: str, category: str = 'custom') -> None:
        """添加自定义指标"""
        metric = MetricData(name, value, unit, datetime.now(), category)
        self.data_manager.save_metric(metric)
        self.logger.info(f"添加自定义指标: {name} = {value} {unit}")
    
    def configure_alert(self, rule_name: str, **kwargs) -> bool:
        """配置告警规则"""
        for rule in self.alert_manager.rules:
            if rule.name == rule_name:
                for key, value in kwargs.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                self.logger.info(f"更新告警规则: {rule_name}")
                return True
        return False