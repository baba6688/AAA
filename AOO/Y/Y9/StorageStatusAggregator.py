#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y9存储状态聚合器

一个综合的存储状态聚合器，用于收集、分析和监控多个存储模块的状态。
提供实时监控、预警机制、历史记录和可视化仪表板功能。

作者: Y9存储团队
版本: 1.0.0
日期: 2025-11-06
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
import statistics
from pathlib import Path


class StorageStatus(Enum):
    """存储状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class AlertLevel(Enum):
    """预警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class StorageModule:
    """存储模块信息"""
    module_id: str
    name: str
    storage_type: str
    capacity: float
    used_space: float
    status: StorageStatus
    response_time: float
    last_update: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def usage_percentage(self) -> float:
        """计算使用率百分比"""
        if self.capacity == 0:
            return 0.0
        return (self.used_space / self.capacity) * 100

    @property
    def available_space(self) -> float:
        """计算可用空间"""
        return self.capacity - self.used_space


@dataclass
class Alert:
    """预警信息"""
    alert_id: str
    module_id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StorageReport:
    """存储状态报告"""
    report_id: str
    timestamp: datetime
    total_modules: int
    healthy_modules: int
    warning_modules: int
    critical_modules: int
    offline_modules: int
    total_capacity: float
    total_used: float
    average_usage: float
    alerts: List[Alert]
    trends: Dict[str, Any]

    @property
    def overall_health_score(self) -> float:
        """计算整体健康分数"""
        if self.total_modules == 0:
            return 0.0
        
        weights = {
            StorageStatus.HEALTHY: 1.0,
            StorageStatus.WARNING: 0.7,
            StorageStatus.CRITICAL: 0.3,
            StorageStatus.OFFLINE: 0.0,
            StorageStatus.MAINTENANCE: 0.5
        }
        
        total_score = 0.0
        module_counts = {
            StorageStatus.HEALTHY: self.healthy_modules,
            StorageStatus.WARNING: self.warning_modules,
            StorageStatus.CRITICAL: self.critical_modules,
            StorageStatus.OFFLINE: self.offline_modules,
            StorageStatus.MAINTENANCE: 0
        }
        
        for status, count in module_counts.items():
            total_score += weights[status] * count
        
        return (total_score / self.total_modules) * 100


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str = "storage_status.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建存储模块表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS storage_modules (
                    module_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    storage_type TEXT NOT NULL,
                    capacity REAL NOT NULL,
                    used_space REAL NOT NULL,
                    status TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    last_update TIMESTAMP NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # 创建预警表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    module_id TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    metadata TEXT,
                    FOREIGN KEY (module_id) REFERENCES storage_modules (module_id)
                )
            ''')
            
            # 创建历史记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    capacity REAL NOT NULL,
                    used_space REAL NOT NULL,
                    status TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    FOREIGN KEY (module_id) REFERENCES storage_modules (module_id)
                )
            ''')
            
            conn.commit()
    
    def save_storage_module(self, module: StorageModule):
        """保存存储模块信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO storage_modules 
                (module_id, name, storage_type, capacity, used_space, status, 
                 response_time, last_update, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                module.module_id, module.name, module.storage_type,
                module.capacity, module.used_space, module.status.value,
                module.response_time, module.last_update,
                json.dumps(module.metadata)
            ))
            conn.commit()
    
    def get_storage_modules(self) -> List[StorageModule]:
        """获取所有存储模块信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM storage_modules')
            rows = cursor.fetchall()
            
            modules = []
            for row in rows:
                module = StorageModule(
                    module_id=row[0],
                    name=row[1],
                    storage_type=row[2],
                    capacity=row[3],
                    used_space=row[4],
                    status=StorageStatus(row[5]),
                    response_time=row[6],
                    last_update=datetime.fromisoformat(row[7]),
                    metadata=json.loads(row[8]) if row[8] else {}
                )
                modules.append(module)
            
            return modules
    
    def save_alert(self, alert: Alert):
        """保存预警信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO alerts 
                (alert_id, module_id, level, message, timestamp, resolved, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id, alert.module_id, alert.level.value,
                alert.message, alert.timestamp, 1 if alert.resolved else 0,
                json.dumps(alert.metadata)
            ))
            conn.commit()
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃预警"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM alerts WHERE resolved = 0')
            rows = cursor.fetchall()
            
            alerts = []
            for row in rows:
                alert = Alert(
                    alert_id=row[0],
                    module_id=row[1],
                    level=AlertLevel(row[2]),
                    message=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    resolved=bool(row[5]),
                    metadata=json.loads(row[6]) if row[6] else {}
                )
                alerts.append(alert)
            
            return alerts
    
    def save_history(self, module: StorageModule):
        """保存历史记录"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO history 
                (module_id, timestamp, capacity, used_space, status, response_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                module.module_id, module.last_update, module.capacity,
                module.used_space, module.status.value, module.response_time
            ))
            conn.commit()
    
    def get_history(self, module_id: str, days: int = 7) -> List[Dict]:
        """获取历史记录"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            start_date = datetime.now() - timedelta(days=days)
            cursor.execute('''
                SELECT * FROM history 
                WHERE module_id = ? AND timestamp >= ?
                ORDER BY timestamp
            ''', (module_id, start_date.isoformat()))
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                try:
                    # 尝试解析时间戳
                    if isinstance(row[1], str):
                        timestamp = datetime.fromisoformat(row[1])
                    else:
                        timestamp = row[1]
                except (ValueError, TypeError):
                    # 如果解析失败，使用当前时间
                    timestamp = datetime.now()
                
                # 确保数值类型正确
                try:
                    capacity = float(row[2]) if row[2] is not None else 0.0
                    used_space = float(row[3]) if row[3] is not None else 0.0
                    response_time = float(row[5]) if row[5] is not None else 0.0
                except (ValueError, TypeError):
                    capacity = 0.0
                    used_space = 0.0
                    response_time = 0.0
                
                history.append({
                    'timestamp': timestamp,
                    'capacity': capacity,
                    'used_space': used_space,
                    'status': str(row[4]) if row[4] is not None else "unknown",
                    'response_time': response_time
                })
            
            return history


class StatusCollector:
    """状态收集器"""
    
    def __init__(self, aggregator: 'StorageStatusAggregator'):
        self.aggregator = aggregator
        self.collectors = {}
        self.running = False
        self.thread = None
    
    def register_collector(self, name: str, collector_func: Callable):
        """注册状态收集器"""
        self.collectors[name] = collector_func
    
    def start_collection(self, interval: int = 30):
        """开始状态收集"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collection_loop, args=(interval,))
        self.thread.daemon = True
        self.thread.start()
    
    def stop_collection(self):
        """停止状态收集"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _collection_loop(self, interval: int):
        """收集循环"""
        while self.running:
            try:
                for name, collector_func in self.collectors.items():
                    try:
                        modules = collector_func()
                        for module in modules:
                            self.aggregator.update_storage_module(module)
                    except Exception as e:
                        logging.error(f"收集器 {name} 执行失败: {e}")
                
                time.sleep(interval)
            except Exception as e:
                logging.error(f"状态收集循环错误: {e}")
                time.sleep(interval)


class AlertManager:
    """预警管理器"""
    
    def __init__(self, aggregator: 'StorageStatusAggregator'):
        self.aggregator = aggregator
        self.alert_rules = []
        self.alert_handlers = []
    
    def add_alert_rule(self, rule_func: Callable[[StorageModule], Optional[Alert]]):
        """添加预警规则"""
        self.alert_rules.append(rule_func)
    
    def add_alert_handler(self, handler_func: Callable[[Alert], None]):
        """添加预警处理器"""
        self.alert_handlers.append(handler_func)
    
    def check_alerts(self, module: StorageModule):
        """检查预警"""
        for rule in self.alert_rules:
            try:
                alert = rule(module)
                if alert:
                    self.aggregator.db_manager.save_alert(alert)
                    for handler in self.alert_handlers:
                        try:
                            handler(alert)
                        except Exception as e:
                            logging.error(f"预警处理器执行失败: {e}")
            except Exception as e:
                logging.error(f"预警规则执行失败: {e}")
    
    def resolve_alert(self, alert_id: str):
        """解决预警"""
        alerts = self.aggregator.db_manager.get_active_alerts()
        for alert in alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                self.aggregator.db_manager.save_alert(alert)
                break


class TrendAnalyzer:
    """趋势分析器"""
    
    def __init__(self, aggregator: 'StorageStatusAggregator'):
        self.aggregator = aggregator
    
    def analyze_usage_trend(self, module_id: str, days: int = 7) -> Dict[str, Any]:
        """分析使用趋势"""
        history = self.aggregator.db_manager.get_history(module_id, days)
        
        if len(history) < 2:
            return {"trend": "insufficient_data", "growth_rate": 0.0}
        
        usage_percentages = [
            (record['used_space'] / record['capacity']) * 100 
            for record in history if record['capacity'] > 0
        ]
        
        if len(usage_percentages) < 2:
            return {"trend": "insufficient_data", "growth_rate": 0.0}
        
        # 计算增长率
        growth_rate = (usage_percentages[-1] - usage_percentages[0]) / len(usage_percentages)
        
        # 判断趋势
        if growth_rate > 1:
            trend = "increasing_fast"
        elif growth_rate > 0.1:
            trend = "increasing_slow"
        elif growth_rate < -1:
            trend = "decreasing_fast"
        elif growth_rate < -0.1:
            trend = "decreasing_slow"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "growth_rate": growth_rate,
            "current_usage": usage_percentages[-1] if usage_percentages else 0,
            "average_usage": statistics.mean(usage_percentages),
            "max_usage": max(usage_percentages),
            "min_usage": min(usage_percentages)
        }
    
    def analyze_performance_trend(self, module_id: str, days: int = 7) -> Dict[str, Any]:
        """分析性能趋势"""
        history = self.aggregator.db_manager.get_history(module_id, days)
        
        response_times = [record['response_time'] for record in history]
        
        if len(response_times) < 2:
            return {"trend": "insufficient_data", "avg_response_time": 0.0}
        
        return {
            "trend": "stable" if statistics.stdev(response_times) < 100 else "variable",
            "avg_response_time": statistics.mean(response_times),
            "max_response_time": max(response_times),
            "min_response_time": min(response_times),
            "response_time_trend": "improving" if response_times[-1] < response_times[0] else "degrading"
        }


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, aggregator: 'StorageStatusAggregator'):
        self.aggregator = aggregator
    
    def generate_comprehensive_report(self) -> StorageReport:
        """生成综合报告"""
        modules = self.aggregator.get_all_modules()
        
        # 统计各状态模块数量
        status_counts = {
            StorageStatus.HEALTHY: 0,
            StorageStatus.WARNING: 0,
            StorageStatus.CRITICAL: 0,
            StorageStatus.OFFLINE: 0,
            StorageStatus.MAINTENANCE: 0
        }
        
        total_capacity = 0
        total_used = 0
        usage_percentages = []
        
        for module in modules:
            status_counts[module.status] += 1
            total_capacity += module.capacity
            total_used += module.used_space
            usage_percentages.append(module.usage_percentage)
        
        # 获取活跃预警
        alerts = self.aggregator.db_manager.get_active_alerts()
        
        # 分析趋势
        trends = {}
        for module in modules:
            trends[module.module_id] = {
                "usage": self.aggregator.trend_analyzer.analyze_usage_trend(module.module_id),
                "performance": self.aggregator.trend_analyzer.analyze_performance_trend(module.module_id)
            }
        
        report = StorageReport(
            report_id=f"report_{int(time.time())}",
            timestamp=datetime.now(),
            total_modules=len(modules),
            healthy_modules=status_counts[StorageStatus.HEALTHY],
            warning_modules=status_counts[StorageStatus.WARNING],
            critical_modules=status_counts[StorageStatus.CRITICAL],
            offline_modules=status_counts[StorageStatus.OFFLINE],
            total_capacity=total_capacity,
            total_used=total_used,
            average_usage=statistics.mean(usage_percentages) if usage_percentages else 0,
            alerts=alerts,
            trends=trends
        )
        
        return report
    
    def generate_json_report(self, report: StorageReport) -> str:
        """生成JSON格式报告"""
        report_dict = asdict(report)
        
        # 转换datetime对象为字符串
        report_dict['timestamp'] = report.timestamp.isoformat()
        for alert in report_dict['alerts']:
            alert['timestamp'] = alert['timestamp'].isoformat()
            # 转换AlertLevel为字符串
            if hasattr(alert['level'], 'value'):
                alert['level'] = alert['level'].value
        
        return json.dumps(report_dict, ensure_ascii=False, indent=2)
    
    def generate_html_dashboard(self, report: StorageReport) -> str:
        """生成HTML仪表板"""
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Y9存储状态仪表板</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                    gap: 20px; margin-bottom: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 10px; 
                       box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .status-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                       gap: 20px; }}
        .status-card {{ background: white; padding: 15px; border-radius: 10px; 
                       box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .status-healthy {{ border-left: 5px solid #4CAF50; }}
        .status-warning {{ border-left: 5px solid #FF9800; }}
        .status-critical {{ border-left: 5px solid #F44336; }}
        .status-offline {{ border-left: 5px solid #9E9E9E; }}
        .alerts {{ background: white; padding: 20px; border-radius: 10px; 
                  box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .alert {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .alert-info {{ background-color: #E3F2FD; border-left: 4px solid #2196F3; }}
        .alert-warning {{ background-color: #FFF3E0; border-left: 4px solid #FF9800; }}
        .alert-error {{ background-color: #FFEBEE; border-left: 4px solid #F44336; }}
        .alert-critical {{ background-color: #F3E5F5; border-left: 4px solid #9C27B0; }}
        .progress-bar {{ background-color: #e0e0e0; border-radius: 10px; overflow: hidden; 
                        height: 20px; margin: 10px 0; }}
        .progress-fill {{ height: 100%; transition: width 0.3s ease; }}
        .progress-healthy {{ background-color: #4CAF50; }}
        .progress-warning {{ background-color: #FF9800; }}
        .progress-critical {{ background-color: #F44336; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Y9存储状态仪表板</h1>
            <p class="timestamp">更新时间: {timestamp}</p>
            <p>整体健康分数: <strong>{health_score:.1f}%</strong></p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{total_modules}</div>
                <div class="metric-label">总模块数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_capacity:.1f} GB</div>
                <div class="metric-label">总容量</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_used:.1f} GB</div>
                <div class="metric-label">已使用</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{average_usage:.1f}%</div>
                <div class="metric-label">平均使用率</div>
            </div>
        </div>
        
        <div class="status-grid">
            {status_cards}
        </div>
        
        <div class="alerts">
            <h2>活跃预警 ({alert_count})</h2>
            {alert_list}
        </div>
    </div>
</body>
</html>
        """
        
        # 生成状态卡片
        status_cards = ""
        for module in self.aggregator.get_all_modules():
            status_class = f"status-{module.status.value}"
            progress_class = "progress-healthy"
            if module.usage_percentage > 80:
                progress_class = "progress-critical"
            elif module.usage_percentage > 60:
                progress_class = "progress-warning"
            
            status_cards += f"""
            <div class="status-card {status_class}">
                <h3>{module.name}</h3>
                <p><strong>类型:</strong> {module.storage_type}</p>
                <p><strong>状态:</strong> {module.status.value}</p>
                <p><strong>响应时间:</strong> {module.response_time:.2f}ms</p>
                <div class="progress-bar">
                    <div class="progress-fill {progress_class}" style="width: {module.usage_percentage:.1f}%"></div>
                </div>
                <p><strong>使用率:</strong> {module.usage_percentage:.1f}% ({module.used_space:.1f}GB / {module.capacity:.1f}GB)</p>
                <p class="timestamp">最后更新: {module.last_update.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """
        
        # 生成预警列表
        alert_list = ""
        for alert in report.alerts:
            alert_class = f"alert-{alert.level.value}"
            alert_list += f"""
            <div class="alert {alert_class}">
                <strong>{alert.level.value.upper()}</strong> - {alert.message}
                <br><small>模块: {alert.module_id} | 时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
            """
        
        if not report.alerts:
            alert_list = "<p>暂无活跃预警</p>"
        
        return html_template.format(
            timestamp=report.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            health_score=report.overall_health_score,
            total_modules=report.total_modules,
            total_capacity=report.total_capacity,
            total_used=report.total_used,
            average_usage=report.average_usage,
            status_cards=status_cards,
            alert_count=len(report.alerts),
            alert_list=alert_list
        )


class StorageStatusAggregator:
    """存储状态聚合器主类"""
    
    def __init__(self, db_path: str = "storage_status.db"):
        self.db_manager = DatabaseManager(db_path)
        self.status_collector = StatusCollector(self)
        self.alert_manager = AlertManager(self)
        self.trend_analyzer = TrendAnalyzer(self)
        self.report_generator = ReportGenerator(self)
        
        # 设置默认预警规则
        self._setup_default_alert_rules()
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_default_alert_rules(self):
        """设置默认预警规则"""
        
        # 高使用率预警
        def high_usage_rule(module: StorageModule) -> Optional[Alert]:
            if module.usage_percentage > 90:
                return Alert(
                    alert_id=f"high_usage_{module.module_id}_{int(time.time())}",
                    module_id=module.module_id,
                    level=AlertLevel.CRITICAL,
                    message=f"存储模块 {module.name} 使用率过高: {module.usage_percentage:.1f}%",
                    timestamp=datetime.now()
                )
            elif module.usage_percentage > 80:
                return Alert(
                    alert_id=f"warning_usage_{module.module_id}_{int(time.time())}",
                    module_id=module.module_id,
                    level=AlertLevel.WARNING,
                    message=f"存储模块 {module.name} 使用率警告: {module.usage_percentage:.1f}%",
                    timestamp=datetime.now()
                )
            return None
        
        # 响应时间预警
        def slow_response_rule(module: StorageModule) -> Optional[Alert]:
            if module.response_time > 5000:
                return Alert(
                    alert_id=f"slow_response_{module.module_id}_{int(time.time())}",
                    module_id=module.module_id,
                    level=AlertLevel.ERROR,
                    message=f"存储模块 {module.name} 响应时间过长: {module.response_time:.2f}ms",
                    timestamp=datetime.now()
                )
            return None
        
        # 离线预警
        def offline_rule(module: StorageModule) -> Optional[Alert]:
            if module.status == StorageStatus.OFFLINE:
                return Alert(
                    alert_id=f"offline_{module.module_id}_{int(time.time())}",
                    module_id=module.module_id,
                    level=AlertLevel.CRITICAL,
                    message=f"存储模块 {module.name} 已离线",
                    timestamp=datetime.now()
                )
            return None
        
        self.alert_manager.add_alert_rule(high_usage_rule)
        self.alert_manager.add_alert_rule(slow_response_rule)
        self.alert_manager.add_alert_rule(offline_rule)
    
    def update_storage_module(self, module: StorageModule):
        """更新存储模块信息"""
        # 保存到数据库
        self.db_manager.save_storage_module(module)
        self.db_manager.save_history(module)
        
        # 检查预警
        self.alert_manager.check_alerts(module)
        
        self.logger.info(f"已更新存储模块: {module.name}")
    
    def get_all_modules(self) -> List[StorageModule]:
        """获取所有存储模块"""
        return self.db_manager.get_storage_modules()
    
    def get_module(self, module_id: str) -> Optional[StorageModule]:
        """获取特定存储模块"""
        modules = self.db_manager.get_storage_modules()
        for module in modules:
            if module.module_id == module_id:
                return module
        return None
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃预警"""
        return self.db_manager.get_active_alerts()
    
    def start_monitoring(self, interval: int = 30):
        """开始监控"""
        self.status_collector.start_collection(interval)
        self.logger.info("开始存储状态监控")
    
    def stop_monitoring(self):
        """停止监控"""
        self.status_collector.stop_collection()
        self.logger.info("停止存储状态监控")
    
    def generate_report(self) -> StorageReport:
        """生成综合报告"""
        return self.report_generator.generate_comprehensive_report()
    
    def export_report_json(self, filepath: str):
        """导出JSON报告"""
        report = self.generate_report()
        json_report = self.report_generator.generate_json_report(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_report)
        
        self.logger.info(f"JSON报告已导出到: {filepath}")
    
    def export_dashboard_html(self, filepath: str):
        """导出HTML仪表板"""
        report = self.generate_report()
        html_dashboard = self.report_generator.generate_html_dashboard(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_dashboard)
        
        self.logger.info(f"HTML仪表板已导出到: {filepath}")
    
    def get_usage_statistics(self, days: int = 7) -> Dict[str, Any]:
        """获取使用统计"""
        modules = self.get_all_modules()
        stats = {}
        
        for module in modules:
            usage_trend = self.trend_analyzer.analyze_usage_trend(module.module_id, days)
            performance_trend = self.trend_analyzer.analyze_performance_trend(module.module_id, days)
            
            stats[module.module_id] = {
                "name": module.name,
                "current_usage": module.usage_percentage,
                "usage_trend": usage_trend,
                "performance_trend": performance_trend
            }
        
        return stats


# 示例使用函数
def create_sample_collector():
    """创建示例状态收集器"""
    def sample_collector():
        """示例收集器函数"""
        modules = [
            StorageModule(
                module_id="module_001",
                name="主存储阵列",
                storage_type="SSD",
                capacity=1000.0,
                used_space=650.0,
                status=StorageStatus.HEALTHY,
                response_time=120.5,
                last_update=datetime.now(),
                metadata={"location": "数据中心A", "vendor": "厂商X"}
            ),
            StorageModule(
                module_id="module_002",
                name="备份存储",
                storage_type="HDD",
                capacity=2000.0,
                used_space=1800.0,
                status=StorageStatus.WARNING,
                response_time=350.2,
                last_update=datetime.now(),
                metadata={"location": "数据中心B", "vendor": "厂商Y"}
            ),
            StorageModule(
                module_id="module_003",
                name="缓存存储",
                storage_type="NVMe",
                capacity=500.0,
                used_space=450.0,
                status=StorageStatus.CRITICAL,
                response_time=1200.8,
                last_update=datetime.now(),
                metadata={"location": "数据中心A", "vendor": "厂商Z"}
            )
        ]
        return modules
    
    return sample_collector


if __name__ == "__main__":
    # 创建聚合器实例
    aggregator = StorageStatusAggregator()
    
    # 注册示例收集器
    sample_collector = create_sample_collector()
    aggregator.status_collector.register_collector("sample", sample_collector)
    
    # 添加预警处理器
    def alert_handler(alert: Alert):
        print(f"预警: [{alert.level.value.upper()}] {alert.message}")
    
    aggregator.alert_manager.add_alert_handler(alert_handler)
    
    # 开始监控
    aggregator.start_monitoring(interval=10)
    
    try:
        # 运行一段时间
        time.sleep(15)
        
        # 生成报告
        report = aggregator.generate_report()
        print(f"\n=== 存储状态报告 ===")
        print(f"报告时间: {report.timestamp}")
        print(f"总模块数: {report.total_modules}")
        print(f"健康模块: {report.healthy_modules}")
        print(f"警告模块: {report.warning_modules}")
        print(f"严重模块: {report.critical_modules}")
        print(f"离线模块: {report.offline_modules}")
        print(f"总容量: {report.total_capacity:.1f} GB")
        print(f"已使用: {report.total_used:.1f} GB")
        print(f"平均使用率: {report.average_usage:.1f}%")
        print(f"整体健康分数: {report.overall_health_score:.1f}%")
        
        # 导出报告
        aggregator.export_report_json("storage_report.json")
        aggregator.export_dashboard_html("storage_dashboard.html")
        
        # 获取统计信息
        stats = aggregator.get_usage_statistics()
        print(f"\n=== 使用统计 ===")
        for module_id, stat in stats.items():
            print(f"{stat['name']}: 当前使用率 {stat['current_usage']:.1f}%, "
                  f"趋势 {stat['usage_trend']['trend']}")
        
    finally:
        # 停止监控
        aggregator.stop_monitoring()