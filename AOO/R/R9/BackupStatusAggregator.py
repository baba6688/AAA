"""
R9备份状态聚合器

该模块提供备份状态的收集、聚合、分析、报告生成和监控功能。
支持多个备份模块的状态聚合，并提供实时监控和预警机制。
"""

import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import os
from pathlib import Path


class BackupStatus(Enum):
    """备份状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    RUNNING = "running"
    PENDING = "pending"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class AlertLevel(Enum):
    """预警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class BackupTaskInfo:
    """备份任务信息"""
    task_id: str
    module_name: str
    status: BackupStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    file_count: int = 0
    total_size: int = 0
    error_message: Optional[str] = None
    progress: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BackupModuleStatus:
    """备份模块状态"""
    module_name: str
    is_active: bool
    last_backup_time: Optional[datetime] = None
    success_rate: float = 0.0
    total_tasks: int = 0
    failed_tasks: int = 0
    average_duration: float = 0.0
    current_status: Optional[BackupStatus] = None


@dataclass
class AlertInfo:
    """预警信息"""
    alert_id: str
    level: AlertLevel
    message: str
    module_name: Optional[str] = None
    task_id: Optional[str] = None
    timestamp: datetime = None
    resolved: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str = "backup_status.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backup_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE,
                    module_name TEXT,
                    status TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    file_count INTEGER,
                    total_size INTEGER,
                    error_message TEXT,
                    progress REAL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE,
                    level TEXT,
                    message TEXT,
                    module_name TEXT,
                    task_id TEXT,
                    timestamp TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS module_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module_name TEXT UNIQUE,
                    is_active BOOLEAN,
                    last_backup_time TEXT,
                    success_rate REAL,
                    total_tasks INTEGER,
                    failed_tasks INTEGER,
                    average_duration REAL,
                    current_status TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def save_backup_task(self, task_info: BackupTaskInfo):
        """保存备份任务信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO backup_tasks 
                (task_id, module_name, status, start_time, end_time, 
                 file_count, total_size, error_message, progress, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_info.task_id,
                task_info.module_name,
                task_info.status.value if isinstance(task_info.status, BackupStatus) else str(task_info.status),
                task_info.start_time.isoformat(),
                task_info.end_time.isoformat() if task_info.end_time else None,
                task_info.file_count,
                task_info.total_size,
                task_info.error_message,
                task_info.progress,
                json.dumps(task_info.metadata) if task_info.metadata else None
            ))
    
    def save_alert(self, alert: AlertInfo):
        """保存预警信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alerts 
                (alert_id, level, message, module_name, task_id, timestamp, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.level.value if isinstance(alert.level, AlertLevel) else str(alert.level),
                alert.message,
                alert.module_name,
                alert.task_id,
                alert.timestamp.isoformat(),
                alert.resolved
            ))
    
    def save_module_status(self, status: BackupModuleStatus):
        """保存模块状态"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO module_status 
                (module_name, is_active, last_backup_time, success_rate,
                 total_tasks, failed_tasks, average_duration, current_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                status.module_name,
                status.is_active,
                status.last_backup_time.isoformat() if status.last_backup_time else None,
                status.success_rate,
                status.total_tasks,
                status.failed_tasks,
                status.average_duration,
                status.current_status.value if isinstance(status.current_status, BackupStatus) else str(status.current_status) if status.current_status else None
            ))


class StatusCollector:
    """状态收集器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.collectors: Dict[str, Callable] = {}
    
    def register_collector(self, module_name: str, collector_func: Callable):
        """注册状态收集器"""
        self.collectors[module_name] = collector_func
    
    def collect_status(self, module_name: str) -> Optional[BackupTaskInfo]:
        """收集指定模块的状态"""
        if module_name in self.collectors:
            try:
                status_data = self.collectors[module_name]()
                if status_data:
                    # 确保status字段是BackupStatus枚举
                    if isinstance(status_data.get('status'), str):
                        status_data['status'] = BackupStatus(status_data['status'])
                    return BackupTaskInfo(**status_data)
            except Exception as e:
                logging.error(f"收集模块 {module_name} 状态失败: {e}")
        return None
    
    def collect_all_status(self) -> List[BackupTaskInfo]:
        """收集所有模块的状态"""
        tasks = []
        for module_name in self.collectors:
            task_info = self.collect_status(module_name)
            if task_info:
                tasks.append(task_info)
        return tasks


class DataAggregator:
    """数据聚合器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def aggregate_by_module(self, days: int = 7) -> Dict[str, BackupModuleStatus]:
        """按模块聚合数据"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT module_name, status, start_time, end_time, error_message
                FROM backup_tasks 
                WHERE start_time >= ? AND start_time <= ?
                ORDER BY module_name, start_time
            """, (start_time.isoformat(), end_time.isoformat()))
            
            module_stats = {}
            for row in cursor.fetchall():
                module_name, status, start_time_str, end_time_str, error_msg = row
                
                if module_name not in module_stats:
                    module_stats[module_name] = {
                        'total_tasks': 0,
                        'failed_tasks': 0,
                        'success_tasks': 0,
                        'total_duration': 0.0,
                        'last_backup_time': None,
                        'current_status': None
                    }
                
                stats = module_stats[module_name]
                stats['total_tasks'] += 1
                
                if status == 'failed':
                    stats['failed_tasks'] += 1
                elif status == 'success':
                    stats['success_tasks'] += 1
                
                # 计算持续时间
                if start_time_str and end_time_str:
                    start_time = datetime.fromisoformat(start_time_str)
                    end_time = datetime.fromisoformat(end_time_str)
                    duration = (end_time - start_time).total_seconds()
                    stats['total_duration'] += duration
                
                # 更新最后备份时间
                if start_time_str:
                    backup_time = datetime.fromisoformat(start_time_str)
                    if not stats['last_backup_time'] or backup_time > stats['last_backup_time']:
                        stats['last_backup_time'] = backup_time
                
                # 更新当前状态
                stats['current_status'] = BackupStatus(status)
            
            # 转换为BackupModuleStatus对象
            result = {}
            for module_name, stats in module_stats.items():
                success_rate = (stats['success_tasks'] / stats['total_tasks'] * 100) if stats['total_tasks'] > 0 else 0.0
                avg_duration = stats['total_duration'] / stats['total_tasks'] if stats['total_tasks'] > 0 else 0.0
                
                result[module_name] = BackupModuleStatus(
                    module_name=module_name,
                    is_active=True,
                    last_backup_time=stats['last_backup_time'],
                    success_rate=success_rate,
                    total_tasks=stats['total_tasks'],
                    failed_tasks=stats['failed_tasks'],
                    average_duration=avg_duration,
                    current_status=stats['current_status']
                )
            
            return result
    
    def get_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """获取趋势分析"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        with sqlite3.connect(self.db_manager.db_path) as conn:
            # 按日期统计
            cursor = conn.execute("""
                SELECT DATE(start_time) as date, status, COUNT(*) as count
                FROM backup_tasks 
                WHERE start_time >= ? AND start_time <= ?
                GROUP BY DATE(start_time), status
                ORDER BY date
            """, (start_time.isoformat(), end_time.isoformat()))
            
            daily_stats = {}
            for row in cursor.fetchall():
                date, status, count = row
                if date not in daily_stats:
                    daily_stats[date] = {'success': 0, 'failed': 0, 'total': 0}
                
                daily_stats[date][status] = count
                daily_stats[date]['total'] += count
            
            # 计算趋势
            success_rates = []
            for date, stats in daily_stats.items():
                if stats['total'] > 0:
                    success_rate = stats['success'] / stats['total'] * 100
                    success_rates.append(success_rate)
            
            trend = "stable"
            if len(success_rates) >= 2:
                recent_avg = sum(success_rates[-7:]) / min(7, len(success_rates))
                older_avg = sum(success_rates[:-7]) / max(1, len(success_rates) - 7)
                
                if recent_avg > older_avg + 5:
                    trend = "improving"
                elif recent_avg < older_avg - 5:
                    trend = "declining"
            
            return {
                'daily_stats': daily_stats,
                'overall_success_rate': sum(success_rates) / len(success_rates) if success_rates else 0.0,
                'trend': trend,
                'period_days': days
            }


class AlertManager:
    """预警管理器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.alert_handlers: List[Callable] = []
        self.alert_rules = []
    
    def add_alert_handler(self, handler: Callable):
        """添加预警处理器"""
        self.alert_handlers.append(handler)
    
    def add_alert_rule(self, rule_func: Callable, condition: Dict[str, Any]):
        """添加预警规则"""
        self.alert_rules.append((rule_func, condition))
    
    def check_and_generate_alerts(self, task_info: BackupTaskInfo) -> List[AlertInfo]:
        """检查并生成预警"""
        alerts = []
        
        # 备份失败预警
        if task_info.status == BackupStatus.FAILED:
            alert = AlertInfo(
                alert_id=f"backup_failed_{task_info.task_id}",
                level=AlertLevel.ERROR,
                message=f"备份任务 {task_info.task_id} 失败: {task_info.error_message}",
                module_name=task_info.module_name,
                task_id=task_info.task_id
            )
            alerts.append(alert)
        
        # 长时间运行预警
        if task_info.status == BackupStatus.RUNNING:
            runtime = (datetime.now() - task_info.start_time).total_seconds()
            if runtime > 4 * 3600:  # 4小时
                alert = AlertInfo(
                    alert_id=f"long_running_{task_info.task_id}",
                    level=AlertLevel.WARNING,
                    message=f"备份任务 {task_info.task_id} 已运行 {runtime/3600:.1f} 小时",
                    module_name=task_info.module_name,
                    task_id=task_info.task_id
                )
                alerts.append(alert)
        
        # 检查自定义规则
        for rule_func, condition in self.alert_rules:
            try:
                if rule_func(task_info, condition):
                    alert = AlertInfo(
                        alert_id=f"rule_{task_info.task_id}_{int(time.time())}",
                        level=AlertLevel.WARNING,
                        message=f"备份任务 {task_info.task_id} 触发预警规则",
                        module_name=task_info.module_name,
                        task_id=task_info.task_id
                    )
                    alerts.append(alert)
            except Exception as e:
                logging.error(f"执行预警规则失败: {e}")
        
        # 保存并处理预警
        for alert in alerts:
            self.db_manager.save_alert(alert)
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logging.error(f"预警处理器执行失败: {e}")
        
        return alerts
    
    def get_recent_alerts(self, hours: int = 24) -> List[AlertInfo]:
        """获取最近的预警"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT alert_id, level, message, module_name, task_id, timestamp, resolved
                FROM alerts 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            """, (start_time.isoformat(), end_time.isoformat()))
            
            alerts = []
            for row in cursor.fetchall():
                alert_id, level, message, module_name, task_id, timestamp, resolved = row
                alerts.append(AlertInfo(
                    alert_id=alert_id,
                    level=AlertLevel(level),
                    message=message,
                    module_name=module_name,
                    task_id=task_id,
                    timestamp=datetime.fromisoformat(timestamp),
                    resolved=bool(resolved)
                ))
            
            return alerts


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, db_manager: DatabaseManager, data_aggregator: DataAggregator):
        self.db_manager = db_manager
        self.data_aggregator = data_aggregator
    
    def generate_summary_report(self, days: int = 7) -> Dict[str, Any]:
        """生成摘要报告"""
        module_stats = self.data_aggregator.aggregate_by_module(days)
        trend_analysis = self.data_aggregator.get_trend_analysis(days)
        
        # 计算总体统计
        total_tasks = sum(stat.total_tasks for stat in module_stats.values())
        total_failures = sum(stat.failed_tasks for stat in module_stats.values())
        overall_success_rate = ((total_tasks - total_failures) / total_tasks * 100) if total_tasks > 0 else 0.0
        
        # 活跃模块统计
        active_modules = [stat for stat in module_stats.values() if stat.is_active]
        
        # 最近备份时间
        recent_backups = [stat.last_backup_time for stat in module_stats.values() if stat.last_backup_time]
        last_backup = max(recent_backups) if recent_backups else None
        
        return {
            'report_period': f'{days} 天',
            'generated_at': datetime.now().isoformat(),
            'overall_stats': {
                'total_tasks': total_tasks,
                'total_failures': total_failures,
                'overall_success_rate': overall_success_rate,
                'active_modules': len(active_modules),
                'total_modules': len(module_stats),
                'last_backup_time': last_backup.isoformat() if last_backup else None
            },
            'module_details': {name: asdict(stat) for name, stat in module_stats.items()},
            'trend_analysis': trend_analysis
        }
    
    def generate_detailed_report(self, module_name: str = None, days: int = 7) -> Dict[str, Any]:
        """生成详细报告"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        with sqlite3.connect(self.db_manager.db_path) as conn:
            query = """
                SELECT task_id, module_name, status, start_time, end_time, 
                       file_count, total_size, error_message, progress, metadata
                FROM backup_tasks 
                WHERE start_time >= ? AND start_time <= ?
            """
            params = [start_time.isoformat(), end_time.isoformat()]
            
            if module_name:
                query += " AND module_name = ?"
                params.append(module_name)
            
            query += " ORDER BY start_time DESC"
            
            cursor = conn.execute(query, params)
            
            tasks = []
            for row in cursor.fetchall():
                task_id, mod_name, status, start_time_str, end_time_str, file_count, total_size, error_msg, progress, metadata = row
                
                task = {
                    'task_id': task_id,
                    'module_name': mod_name,
                    'status': status,
                    'start_time': start_time_str,
                    'end_time': end_time_str,
                    'file_count': file_count,
                    'total_size': total_size,
                    'error_message': error_msg,
                    'progress': progress,
                    'metadata': json.loads(metadata) if metadata else None
                }
                tasks.append(task)
            
            return {
                'report_period': f'{days} 天',
                'module_filter': module_name,
                'generated_at': datetime.now().isoformat(),
                'task_count': len(tasks),
                'tasks': tasks
            }
    
    def export_report(self, report_data: Dict[str, Any], format: str = 'json', file_path: str = None) -> str:
        """导出报告"""
        if not file_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f'backup_report_{timestamp}.{format}'
        
        if format == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        elif format == 'txt':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("R9备份状态聚合器报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"报告时间: {report_data.get('generated_at', 'N/A')}\n")
                f.write(f"报告周期: {report_data.get('report_period', 'N/A')}\n\n")
                
                if 'overall_stats' in report_data:
                    stats = report_data['overall_stats']
                    f.write("总体统计:\n")
                    f.write(f"  总任务数: {stats.get('total_tasks', 0)}\n")
                    f.write(f"  失败任务数: {stats.get('total_failures', 0)}\n")
                    f.write(f"  成功率: {stats.get('overall_success_rate', 0):.2f}%\n")
                    f.write(f"  活跃模块: {stats.get('active_modules', 0)}/{stats.get('total_modules', 0)}\n\n")
        
        return file_path


class StatusMonitor:
    """状态监控器"""
    
    def __init__(self, backup_aggregator):
        self.backup_aggregator = backup_aggregator
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 60  # 60秒检查一次
    
    def start_monitoring(self):
        """开始监控"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logging.info("备份状态监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logging.info("备份状态监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集所有模块状态
                tasks = self.backup_aggregator.status_collector.collect_all_status()
                
                # 检查预警
                for task in tasks:
                    self.backup_aggregator.alert_manager.check_and_generate_alerts(task)
                
                # 更新模块状态
                for task in tasks:
                    self.backup_aggregator._update_module_status(task)
                
                time.sleep(self.monitor_interval)
            except Exception as e:
                logging.error(f"监控循环出错: {e}")
                time.sleep(self.monitor_interval)


class Dashboard:
    """仪表板"""
    
    def __init__(self, backup_aggregator):
        self.backup_aggregator = backup_aggregator
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        # 获取摘要报告
        summary = self.backup_aggregator.report_generator.generate_summary_report(7)
        
        # 获取最近预警
        recent_alerts = self.backup_aggregator.alert_manager.get_recent_alerts(24)
        
        # 获取运行中的任务
        running_tasks = []
        with sqlite3.connect(self.backup_aggregator.db_manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT task_id, module_name, status, start_time, progress
                FROM backup_tasks 
                WHERE status = 'running'
                ORDER BY start_time DESC
            """)
            
            for row in cursor.fetchall():
                running_tasks.append({
                    'task_id': row[0],
                    'module_name': row[1],
                    'status': row[2],
                    'start_time': row[3],
                    'progress': row[4]
                })
        
        return {
            'summary': summary,
            'recent_alerts': [asdict(alert) for alert in recent_alerts],
            'running_tasks': running_tasks,
            'last_updated': datetime.now().isoformat()
        }
    
    def generate_html_dashboard(self, output_path: str = 'backup_dashboard.html'):
        """生成HTML仪表板"""
        data = self.get_dashboard_data()
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>R9备份状态仪表板</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
                .stat-card {{ background-color: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; flex: 1; }}
                .module-status {{ margin: 20px 0; }}
                .module-card {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .status-success {{ border-left: 5px solid #4CAF50; }}
                .status-failed {{ border-left: 5px solid #f44336; }}
                .status-running {{ border-left: 5px solid #2196F3; }}
                .alerts {{ margin: 20px 0; }}
                .alert {{ padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .alert-error {{ background-color: #ffebee; color: #c62828; }}
                .alert-warning {{ background-color: #fff3e0; color: #ef6c00; }}
                .alert-info {{ background-color: #e3f2fd; color: #1565c0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>R9备份状态仪表板</h1>
                <p>最后更新: {last_updated}</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <h3>总体统计</h3>
                    <p>总任务数: {total_tasks}</p>
                    <p>失败任务数: {total_failures}</p>
                    <p>成功率: {success_rate:.2f}%</p>
                    <p>活跃模块: {active_modules}/{total_modules}</p>
                </div>
                <div class="stat-card">
                    <h3>运行状态</h3>
                    <p>运行中任务: {running_tasks}</p>
                    <p>趋势: {trend}</p>
                    <p>最近备份: {last_backup}</p>
                </div>
            </div>
            
            <div class="module-status">
                <h2>模块状态</h2>
                {module_cards}
            </div>
            
            <div class="alerts">
                <h2>最近预警</h2>
                {alert_list}
            </div>
        </body>
        </html>
        """
        
        # 填充数据
        summary = data['summary']
        stats = summary.get('overall_stats', {})
        
        module_cards = ""
        for module_name, module_data in summary.get('module_details', {}).items():
            status_class = f"status-{module_data.get('current_status', 'unknown')}"
            module_cards += f"""
            <div class="module-card {status_class}">
                <h3>{module_name}</h3>
                <p>状态: {module_data.get('current_status', 'unknown')}</p>
                <p>成功率: {module_data.get('success_rate', 0):.2f}%</p>
                <p>总任务: {module_data.get('total_tasks', 0)}</p>
                <p>失败任务: {module_data.get('failed_tasks', 0)}</p>
                <p>平均持续时间: {module_data.get('average_duration', 0):.2f}秒</p>
            </div>
            """
        
        alert_list = ""
        for alert in data['recent_alerts'][:10]:  # 只显示最近10个预警
            alert_class = f"alert-{alert['level']}"
            alert_list += f"""
            <div class="alert {alert_class}">
                <strong>{alert['level'].upper()}</strong> - {alert['message']}
                <br><small>{alert['timestamp']}</small>
            </div>
            """
        
        html_content = html_template.format(
            last_updated=data['last_updated'],
            total_tasks=stats.get('total_tasks', 0),
            total_failures=stats.get('total_failures', 0),
            success_rate=stats.get('overall_success_rate', 0),
            active_modules=stats.get('active_modules', 0),
            total_modules=stats.get('total_modules', 0),
            running_tasks=len(data['running_tasks']),
            trend=summary.get('trend_analysis', {}).get('trend', 'unknown'),
            last_backup=stats.get('last_backup_time', 'N/A'),
            module_cards=module_cards,
            alert_list=alert_list or '<p>暂无预警</p>'
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path


class BackupStatusAggregator:
    """备份状态聚合器主类"""
    
    def __init__(self, db_path: str = "backup_status.db"):
        self.db_manager = DatabaseManager(db_path)
        self.status_collector = StatusCollector(self.db_manager)
        self.data_aggregator = DataAggregator(self.db_manager)
        self.alert_manager = AlertManager(self.db_manager)
        self.report_generator = ReportGenerator(self.db_manager, self.data_aggregator)
        self.monitor = StatusMonitor(self)
        self.dashboard = Dashboard(self)
        
        # 设置默认预警处理器
        self.alert_manager.add_alert_handler(self._default_alert_handler)
        
        # 初始化日志
        logging.basicConfig(level=logging.INFO)
    
    def register_backup_module(self, module_name: str, collector_func: Callable):
        """注册备份模块"""
        self.status_collector.register_collector(module_name, collector_func)
        logging.info(f"已注册备份模块: {module_name}")
    
    def update_backup_status(self, task_info: BackupTaskInfo):
        """更新备份状态"""
        # 保存任务信息
        self.db_manager.save_backup_task(task_info)
        
        # 检查预警
        self.alert_manager.check_and_generate_alerts(task_info)
        
        # 更新模块状态
        self._update_module_status(task_info)
        
        logging.info(f"已更新备份状态: {task_info.task_id} - {task_info.status.value}")
    
    def _update_module_status(self, task_info: BackupTaskInfo):
        """更新模块状态"""
        # 获取该模块的历史统计数据
        module_stats = self.data_aggregator.aggregate_by_module(1)  # 最近1天
        
        if task_info.module_name in module_stats:
            status = module_stats[task_info.module_name]
        else:
            # 创建新的模块状态
            status = BackupModuleStatus(
                module_name=task_info.module_name,
                is_active=True
            )
        
        # 更新状态
        status.current_status = task_info.status
        if task_info.status in [BackupStatus.SUCCESS, BackupStatus.FAILED]:
            status.last_backup_time = task_info.start_time
        
        # 保存状态
        self.db_manager.save_module_status(status)
    
    def get_summary_report(self, days: int = 7) -> Dict[str, Any]:
        """获取摘要报告"""
        return self.report_generator.generate_summary_report(days)
    
    def get_detailed_report(self, module_name: str = None, days: int = 7) -> Dict[str, Any]:
        """获取详细报告"""
        return self.report_generator.generate_detailed_report(module_name, days)
    
    def export_report(self, report_type: str = 'summary', module_name: str = None, 
                     days: int = 7, format: str = 'json', file_path: str = None) -> str:
        """导出报告"""
        if report_type == 'summary':
            report_data = self.get_summary_report(days)
        else:
            report_data = self.get_detailed_report(module_name, days)
        
        return self.report_generator.export_report(report_data, format, file_path)
    
    def start_monitoring(self):
        """开始监控"""
        self.monitor.start_monitoring()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitor.stop_monitoring()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        return self.dashboard.get_dashboard_data()
    
    def generate_html_dashboard(self, output_path: str = None) -> str:
        """生成HTML仪表板"""
        if not output_path:
            output_path = 'backup_dashboard.html'
        return self.dashboard.generate_html_dashboard(output_path)
    
    def get_recent_alerts(self, hours: int = 24) -> List[AlertInfo]:
        """获取最近的预警"""
        return self.alert_manager.get_recent_alerts(hours)
    
    def resolve_alert(self, alert_id: str):
        """解决预警"""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            conn.execute("UPDATE alerts SET resolved = TRUE WHERE alert_id = ?", (alert_id,))
        logging.info(f"已解决预警: {alert_id}")
    
    def _default_alert_handler(self, alert: AlertInfo):
        """默认预警处理器"""
        if alert.level == AlertLevel.CRITICAL:
            logging.critical(f"严重预警: {alert.message}")
        elif alert.level == AlertLevel.ERROR:
            logging.error(f"错误预警: {alert.message}")
        elif alert.level == AlertLevel.WARNING:
            logging.warning(f"警告预警: {alert.message}")
        else:
            logging.info(f"信息预警: {alert.message}")
    
    def cleanup_old_data(self, days: int = 30):
        """清理旧数据"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_manager.db_path) as conn:
            # 清理旧的任务记录
            conn.execute("DELETE FROM backup_tasks WHERE created_at < ?", (cutoff_time.isoformat(),))
            
            # 清理已解决的旧预警
            conn.execute("DELETE FROM alerts WHERE resolved = TRUE AND created_at < ?", (cutoff_time.isoformat(),))
        
        logging.info(f"已清理 {days} 天前的旧数据")


# 示例使用函数
def create_sample_collector(module_name: str):
    """创建示例状态收集器"""
    import random
    
    def collector():
        return {
            'task_id': f'task_{module_name}_{int(time.time())}',
            'module_name': module_name,
            'status': random.choice(['success', 'failed', 'running']),
            'start_time': datetime.now() - timedelta(minutes=random.randint(1, 60)),
            'file_count': random.randint(100, 1000),
            'total_size': random.randint(1000000, 10000000),
            'progress': random.uniform(0, 100)
        }
    
    return collector


if __name__ == "__main__":
    # 示例用法
    aggregator = BackupStatusAggregator()
    
    # 注册示例模块
    modules = ['database', 'files', 'config', 'logs']
    for module in modules:
        aggregator.register_backup_module(module, create_sample_collector(module))
    
    # 启动监控
    aggregator.start_monitoring()
    
    try:
        # 运行一段时间
        time.sleep(10)
        
        # 生成报告
        report = aggregator.get_summary_report()
        print("摘要报告:")
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        
        # 生成仪表板
        dashboard_path = aggregator.generate_html_dashboard()
        print(f"仪表板已生成: {dashboard_path}")
        
    finally:
        aggregator.stop_monitoring()