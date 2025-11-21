#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X9缓存状态聚合器

该模块提供了一个完整的缓存状态聚合系统，包括：
- 状态收集器：从各个缓存模块收集状态信息
- 数据聚合：聚合多个缓存模块的结果
- 状态分析：分析缓存状态和趋势
- 报告生成：生成综合缓存状态报告
- 状态监控：实时监控缓存状态
- 预警机制：缓存异常时预警
- 历史记录：保存历史缓存状态
- 仪表板：提供可视化的缓存状态仪表板

作者: X9开发团队
版本: 1.0.0
日期: 2025-11-06
"""

import json
import time
import threading
import sqlite3
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
import os
import hashlib

# 可选导入matplotlib和pandas
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.animation import FuncAnimation
    import pandas as pd
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("matplotlib或pandas未安装，可视化功能将不可用")


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cache_status_aggregator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class CacheStatus:
    """缓存状态数据类"""
    cache_name: str
    status: str  # 'healthy', 'warning', 'critical', 'unknown'
    hit_rate: float
    miss_rate: float
    memory_usage: int  # bytes
    max_memory: int  # bytes
    entry_count: int
    max_entries: int
    avg_response_time: float  # ms
    last_access_time: datetime
    error_count: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['last_access_time'] = self.last_access_time.isoformat()
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheStatus':
        """从字典创建实例"""
        data['last_access_time'] = datetime.fromisoformat(data['last_access_time'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class AlertRule:
    """预警规则数据类"""
    rule_name: str
    metric: str  # 'hit_rate', 'memory_usage', 'response_time', etc.
    threshold: float
    operator: str  # '>', '<', '>=', '<=', '=='
    severity: str  # 'low', 'medium', 'high', 'critical'
    enabled: bool = True


@dataclass
class Alert:
    """预警数据类"""
    alert_id: str
    rule_name: str
    cache_name: str
    metric: str
    current_value: float
    threshold: float
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class StatusCollector:
    """状态收集器 - 从各个缓存模块收集状态信息"""
    
    def __init__(self):
        self.cache_modules: Dict[str, Any] = {}
        self.collectors: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        
    def register_cache_module(self, name: str, module: Any, collector_func: Callable):
        """注册缓存模块及其收集函数"""
        with self._lock:
            self.cache_modules[name] = module
            self.collectors[name] = collector_func
            logger.info(f"注册缓存模块: {name}")
    
    def collect_status(self, cache_name: str) -> Optional[CacheStatus]:
        """收集指定缓存的状态信息"""
        if cache_name not in self.collectors:
            logger.warning(f"未找到缓存模块: {cache_name}")
            return None
        
        try:
            # 获取对应的缓存模块
            cache_module = self.cache_modules.get(cache_name)
            if cache_module is None:
                logger.warning(f"未找到缓存模块: {cache_name}")
                return None
                
            # 调用收集函数，传递缓存模块
            status_data = self.collectors[cache_name](cache_module)
            return CacheStatus(
                cache_name=cache_name,
                status=status_data.get('status', 'unknown'),
                hit_rate=status_data.get('hit_rate', 0.0),
                miss_rate=status_data.get('miss_rate', 0.0),
                memory_usage=status_data.get('memory_usage', 0),
                max_memory=status_data.get('max_memory', 0),
                entry_count=status_data.get('entry_count', 0),
                max_entries=status_data.get('max_entries', 0),
                avg_response_time=status_data.get('avg_response_time', 0.0),
                last_access_time=datetime.fromisoformat(status_data.get('last_access_time', datetime.now().isoformat())),
                error_count=status_data.get('error_count', 0),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"收集缓存 {cache_name} 状态时出错: {e}")
            return None
    
    def collect_all_status(self) -> List[CacheStatus]:
        """收集所有注册缓存的状态信息"""
        status_list = []
        with self._lock:
            cache_names = list(self.cache_modules.keys())
        
        for cache_name in cache_names:
            status = self.collect_status(cache_name)
            if status:
                status_list.append(status)
        
        return status_list


class DataAggregator:
    """数据聚合器 - 聚合多个缓存模块的结果"""
    
    def __init__(self):
        self.aggregation_rules: Dict[str, str] = {
            'total_memory_usage': 'sum',
            'total_entries': 'sum',
            'avg_hit_rate': 'mean',
            'avg_response_time': 'mean',
            'total_errors': 'sum'
        }
    
    def aggregate_status(self, status_list: List[CacheStatus]) -> Dict[str, Any]:
        """聚合多个缓存状态"""
        if not status_list:
            return {}
        
        aggregated = {
            'total_caches': len(status_list),
            'healthy_caches': len([s for s in status_list if s.status == 'healthy']),
            'warning_caches': len([s for s in status_list if s.status == 'warning']),
            'critical_caches': len([s for s in status_list if s.status == 'critical']),
            'total_memory_usage': sum(s.memory_usage for s in status_list),
            'total_max_memory': sum(s.max_memory for s in status_list),
            'total_entries': sum(s.entry_count for s in status_list),
            'total_max_entries': sum(s.max_entries for s in status_list),
            'avg_hit_rate': statistics.mean(s.hit_rate for s in status_list),
            'avg_miss_rate': statistics.mean(s.miss_rate for s in status_list),
            'avg_response_time': statistics.mean(s.avg_response_time for s in status_list),
            'total_errors': sum(s.error_count for s in status_list),
            'memory_utilization': 0.0,
            'entry_utilization': 0.0,
            'overall_health_score': 0.0
        }
        
        # 计算利用率
        if aggregated['total_max_memory'] > 0:
            aggregated['memory_utilization'] = aggregated['total_memory_usage'] / aggregated['total_max_memory']
        
        if aggregated['total_max_entries'] > 0:
            aggregated['entry_utilization'] = aggregated['total_entries'] / aggregated['total_max_entries']
        
        # 计算整体健康分数
        health_score = 0.0
        if aggregated['total_caches'] > 0:
            health_score = (
                (aggregated['healthy_caches'] * 1.0 + 
                 aggregated['warning_caches'] * 0.7 + 
                 aggregated['critical_caches'] * 0.3) / aggregated['total_caches']
            ) * 100
        
        aggregated['overall_health_score'] = health_score
        
        return aggregated
    
    def aggregate_by_time_range(self, status_history: List[Dict[str, Any]], 
                               time_range: str) -> Dict[str, Any]:
        """按时间范围聚合历史数据"""
        now = datetime.now()
        
        if time_range == 'hour':
            start_time = now - timedelta(hours=1)
        elif time_range == 'day':
            start_time = now - timedelta(days=1)
        elif time_range == 'week':
            start_time = now - timedelta(weeks=1)
        elif time_range == 'month':
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)  # 默认一天
        
        filtered_data = [
            record for record in status_history
            if datetime.fromisoformat(record['timestamp']) >= start_time
        ]
        
        if not filtered_data:
            return {}
        
        # 按小时分组聚合
        hourly_data = defaultdict(list)
        for record in filtered_data:
            timestamp = datetime.fromisoformat(record['timestamp'])
            hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_data[hour_key].append(record)
        
        aggregated_trends = {}
        for hour, records in hourly_data.items():
            aggregated_trends[hour.isoformat()] = self.aggregate_status([
                CacheStatus.from_dict(record) for record in records
            ])
        
        return aggregated_trends


class StatusAnalyzer:
    """状态分析器 - 分析缓存状态和趋势"""
    
    def __init__(self):
        self.trend_analysis_window = 24  # 24小时趋势窗口
        
    def analyze_trends(self, status_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析缓存状态趋势"""
        if len(status_history) < 2:
            return {'trend': 'insufficient_data', 'changes': {}}
        
        # 按时间排序
        sorted_history = sorted(status_history, key=lambda x: x['timestamp'])
        
        # 分析各项指标趋势
        trends = {}
        
        # 分析命中率趋势
        hit_rates = [record['avg_hit_rate'] for record in sorted_history]
        if len(hit_rates) >= 2:
            recent_avg = statistics.mean(hit_rates[-5:])  # 最近5个数据点
            earlier_avg = statistics.mean(hit_rates[:5])   # 前面5个数据点
            
            if recent_avg > earlier_avg * 1.05:
                trends['hit_rate'] = 'improving'
            elif recent_avg < earlier_avg * 0.95:
                trends['hit_rate'] = 'declining'
            else:
                trends['hit_rate'] = 'stable'
        
        # 分析内存使用趋势
        memory_usage = [record['total_memory_usage'] for record in sorted_history]
        if len(memory_usage) >= 2:
            recent_avg = statistics.mean(memory_usage[-5:])
            earlier_avg = statistics.mean(memory_usage[:5])
            
            if recent_avg > earlier_avg * 1.1:
                trends['memory_usage'] = 'increasing'
            elif recent_avg < earlier_avg * 0.9:
                trends['memory_usage'] = 'decreasing'
            else:
                trends['memory_usage'] = 'stable'
        
        # 分析响应时间趋势
        response_times = [record['avg_response_time'] for record in sorted_history]
        if len(response_times) >= 2:
            recent_avg = statistics.mean(response_times[-5:])
            earlier_avg = statistics.mean(response_times[:5])
            
            if recent_avg > earlier_avg * 1.1:
                trends['response_time'] = 'increasing'
            elif recent_avg < earlier_avg * 0.9:
                trends['response_time'] = 'decreasing'
            else:
                trends['response_time'] = 'stable'
        
        return {
            'trend': trends,
            'data_points': len(sorted_history),
            'analysis_period': self._get_analysis_period(sorted_history)
        }
    
    def detect_anomalies(self, current_status: CacheStatus, 
                        historical_data: List[Dict[str, Any]]) -> List[str]:
        """检测异常状态"""
        anomalies = []
        
        if len(historical_data) < 5:
            return anomalies
        
        # 提取历史数据中的相同指标
        historical_hits = [record['avg_hit_rate'] for record in historical_data]
        historical_memory = [record['total_memory_usage'] for record in historical_data]
        historical_response = [record['avg_response_time'] for record in historical_data]
        
        # 检测命中率异常
        if current_status.hit_rate < statistics.mean(historical_hits) - 2 * statistics.stdev(historical_hits):
            anomalies.append(f"命中率异常低: {current_status.hit_rate:.2%}")
        
        # 检测内存使用异常
        if current_status.memory_usage > statistics.mean(historical_memory) + 2 * statistics.stdev(historical_memory):
            anomalies.append(f"内存使用异常高: {current_status.memory_usage} bytes")
        
        # 检测响应时间异常
        if current_status.avg_response_time > statistics.mean(historical_response) + 2 * statistics.stdev(historical_response):
            anomalies.append(f"响应时间异常高: {current_status.avg_response_time:.2f}ms")
        
        return anomalies
    
    def _get_analysis_period(self, history: List[Dict[str, Any]]) -> str:
        """获取分析时间段"""
        if not history:
            return "unknown"
        
        # 确保时间戳是字符串格式
        start_timestamp = history[0]['timestamp']
        end_timestamp = history[-1]['timestamp']
        
        if isinstance(start_timestamp, str):
            start_time = datetime.fromisoformat(start_timestamp)
        else:
            start_time = start_timestamp
            
        if isinstance(end_timestamp, str):
            end_time = datetime.fromisoformat(end_timestamp)
        else:
            end_time = end_timestamp
        
        duration = end_time - start_time
        
        if duration < timedelta(hours=1):
            return "minutes"
        elif duration < timedelta(days=1):
            return "hours"
        elif duration < timedelta(days=7):
            return "days"
        else:
            return "weeks"


class AlertManager:
    """预警管理器 - 缓存异常时预警"""
    
    def __init__(self):
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        self._lock = threading.Lock()
    
    def add_rule(self, rule: AlertRule):
        """添加预警规则"""
        with self._lock:
            self.alert_rules.append(rule)
            logger.info(f"添加预警规则: {rule.rule_name}")
    
    def remove_rule(self, rule_name: str):
        """移除预警规则"""
        with self._lock:
            self.alert_rules = [r for r in self.alert_rules if r.rule_name != rule_name]
            logger.info(f"移除预警规则: {rule_name}")
    
    def check_alerts(self, status: CacheStatus) -> List[Alert]:
        """检查预警条件"""
        new_alerts = []
        
        with self._lock:
            for rule in self.alert_rules:
                if not rule.enabled:
                    continue
                
                current_value = self._get_metric_value(status, rule.metric)
                if current_value is None:
                    continue
                
                if self._evaluate_condition(current_value, rule.threshold, rule.operator):
                    alert = self._create_alert(rule, status, current_value)
                    alert_id = self._generate_alert_id(alert)
                    
                    if alert_id not in self.active_alerts:
                        self.active_alerts[alert_id] = alert
                        new_alerts.append(alert)
                        logger.warning(f"触发预警: {alert.message}")
        
        return new_alerts
    
    def resolve_alert(self, alert_id: str):
        """解决预警"""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                logger.info(f"解决预警: {alert.message}")
    
    def add_alert_callback(self, callback: Callable):
        """添加预警回调函数"""
        with self._lock:
            self.alert_callbacks.append(callback)
    
    def _get_metric_value(self, status: CacheStatus, metric: str) -> Optional[float]:
        """获取指标值"""
        metric_map = {
            'hit_rate': status.hit_rate,
            'miss_rate': status.miss_rate,
            'memory_usage': status.memory_usage,
            'memory_utilization': status.memory_usage / status.max_memory if status.max_memory > 0 else 0,
            'entry_count': status.entry_count,
            'entry_utilization': status.entry_count / status.max_entries if status.max_entries > 0 else 0,
            'response_time': status.avg_response_time,
            'error_count': status.error_count
        }
        return metric_map.get(metric)
    
    def _evaluate_condition(self, value: float, threshold: float, operator: str) -> bool:
        """评估条件"""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return abs(value - threshold) < 1e-6
        else:
            return False
    
    def _create_alert(self, rule: AlertRule, status: CacheStatus, current_value: float) -> Alert:
        """创建预警"""
        message = f"缓存 {status.cache_name} {rule.metric} {rule.operator} {rule.threshold}, 当前值: {current_value}"
        
        return Alert(
            alert_id="",
            rule_name=rule.rule_name,
            cache_name=status.cache_name,
            metric=rule.metric,
            current_value=current_value,
            threshold=rule.threshold,
            severity=rule.severity,
            message=message,
            timestamp=datetime.now()
        )
    
    def _generate_alert_id(self, alert: Alert) -> str:
        """生成预警ID"""
        content = f"{alert.rule_name}_{alert.cache_name}_{alert.metric}_{alert.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class HistoryManager:
    """历史记录管理器 - 保存历史缓存状态"""
    
    def __init__(self, db_path: str = "cache_status_history.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_status_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    hit_rate REAL NOT NULL,
                    miss_rate REAL NOT NULL,
                    memory_usage INTEGER NOT NULL,
                    max_memory INTEGER NOT NULL,
                    entry_count INTEGER NOT NULL,
                    max_entries INTEGER NOT NULL,
                    avg_response_time REAL NOT NULL,
                    last_access_time TEXT NOT NULL,
                    error_count INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_status_history(timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_cache_name ON cache_status_history(cache_name)
            ''')
    
    def save_status(self, status: CacheStatus):
        """保存缓存状态"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO cache_status_history (
                    cache_name, status, hit_rate, miss_rate, memory_usage, max_memory,
                    entry_count, max_entries, avg_response_time, last_access_time,
                    error_count, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                status.cache_name, status.status, status.hit_rate, status.miss_rate,
                status.memory_usage, status.max_memory, status.entry_count,
                status.max_entries, status.avg_response_time,
                status.last_access_time.isoformat(), status.error_count,
                status.timestamp.isoformat()
            ))
    
    def get_history(self, cache_name: Optional[str] = None, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """获取历史记录"""
        query = "SELECT * FROM cache_status_history WHERE 1=1"
        params = []
        
        if cache_name:
            query += " AND cache_name = ?"
            params.append(cache_name)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            history = []
            for row in cursor.fetchall():
                record = dict(row)
                record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                record['last_access_time'] = datetime.fromisoformat(record['last_access_time'])
                history.append(record)
            
            return history
    
    def cleanup_old_records(self, days: int = 30):
        """清理旧记录"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM cache_status_history WHERE timestamp < ?",
                (cutoff_time.isoformat(),)
            )
            deleted_count = cursor.rowcount
            logger.info(f"清理了 {deleted_count} 条超过 {days} 天的记录")
            return deleted_count


class ReportGenerator:
    """报告生成器 - 生成综合缓存状态报告"""
    
    def __init__(self):
        self.report_templates = {
            'summary': self._generate_summary_report,
            'detailed': self._generate_detailed_report,
            'trend': self._generate_trend_report,
            'alert': self._generate_alert_report
        }
    
    def generate_report(self, report_type: str, data: Dict[str, Any]) -> str:
        """生成报告"""
        if report_type not in self.report_templates:
            raise ValueError(f"不支持的报告类型: {report_type}")
        
        return self.report_templates[report_type](data)
    
    def _generate_summary_report(self, data: Dict[str, Any]) -> str:
        """生成概要报告"""
        report = []
        report.append("# X9缓存状态概要报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 整体状态
        report.append("## 整体状态")
        report.append(f"- 缓存总数: {data.get('total_caches', 0)}")
        report.append(f"- 健康缓存: {data.get('healthy_caches', 0)}")
        report.append(f"- 警告缓存: {data.get('warning_caches', 0)}")
        report.append(f"- 严重缓存: {data.get('critical_caches', 0)}")
        report.append(f"- 整体健康分数: {data.get('overall_health_score', 0):.1f}%")
        report.append("")
        
        # 性能指标
        report.append("## 性能指标")
        report.append(f"- 平均命中率: {data.get('avg_hit_rate', 0):.2%}")
        report.append(f"- 平均未命中率: {data.get('avg_miss_rate', 0):.2%}")
        report.append(f"- 平均响应时间: {data.get('avg_response_time', 0):.2f}ms")
        report.append(f"- 总错误数: {data.get('total_errors', 0)}")
        report.append("")
        
        # 资源使用
        report.append("## 资源使用")
        report.append(f"- 总内存使用: {data.get('total_memory_usage', 0) / 1024 / 1024:.2f} MB")
        report.append(f"- 总内存容量: {data.get('total_max_memory', 0) / 1024 / 1024:.2f} MB")
        report.append(f"- 内存利用率: {data.get('memory_utilization', 0):.2%}")
        report.append(f"- 总条目数: {data.get('total_entries', 0)}")
        report.append(f"- 条目利用率: {data.get('entry_utilization', 0):.2%}")
        
        return "\n".join(report)
    
    def _generate_detailed_report(self, data: Dict[str, Any]) -> str:
        """生成详细报告"""
        report = []
        report.append("# X9缓存状态详细报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 概要信息
        report.append("## 概要信息")
        aggregated = data.get('aggregated_status', {})
        for key, value in aggregated.items():
            if isinstance(value, float):
                report.append(f"- {key}: {value:.2f}")
            else:
                report.append(f"- {key}: {value}")
        report.append("")
        
        # 各个缓存详情
        report.append("## 各个缓存详情")
        cache_details = data.get('cache_details', [])
        
        for cache in cache_details:
            report.append(f"### {cache['cache_name']}")
            report.append(f"- 状态: {cache['status']}")
            report.append(f"- 命中率: {cache['hit_rate']:.2%}")
            report.append(f"- 内存使用: {cache['memory_usage'] / 1024 / 1024:.2f} MB")
            report.append(f"- 条目数: {cache['entry_count']}")
            report.append(f"- 响应时间: {cache['avg_response_time']:.2f}ms")
            report.append(f"- 错误数: {cache['error_count']}")
            report.append("")
        
        return "\n".join(report)
    
    def _generate_trend_report(self, data: Dict[str, Any]) -> str:
        """生成趋势报告"""
        report = []
        report.append("# X9缓存状态趋势报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        trends = data.get('trends', {})
        report.append("## 趋势分析")
        
        for metric, trend in trends.items():
            report.append(f"- {metric}: {trend}")
        
        report.append("")
        
        # 趋势数据
        trend_data = data.get('trend_data', {})
        report.append("## 历史趋势数据")
        
        for timestamp, status in trend_data.items():
            report.append(f"### {timestamp}")
            report.append(f"- 整体健康分数: {status.get('overall_health_score', 0):.1f}%")
            report.append(f"- 平均命中率: {status.get('avg_hit_rate', 0):.2%}")
            report.append(f"- 平均响应时间: {status.get('avg_response_time', 0):.2f}ms")
            report.append("")
        
        return "\n".join(report)
    
    def _generate_alert_report(self, data: Dict[str, Any]) -> str:
        """生成预警报告"""
        report = []
        report.append("# X9缓存状态预警报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        active_alerts = data.get('active_alerts', [])
        alert_history = data.get('alert_history', [])
        
        report.append("## 活跃预警")
        if active_alerts:
            for alert in active_alerts:
                report.append(f"- [{alert['severity']}] {alert['message']} ({alert['timestamp']})")
        else:
            report.append("- 无活跃预警")
        
        report.append("")
        
        report.append("## 历史预警")
        if alert_history:
            for alert in alert_history[-10:]:  # 显示最近10条
                status = "已解决" if alert['resolved'] else "未解决"
                report.append(f"- [{alert['severity']}] {alert['message']} - {status}")
        else:
            report.append("- 无历史预警")
        
        return "\n".join(report)


class Dashboard:
    """仪表板 - 提供可视化的缓存状态仪表板"""
    
    def __init__(self, aggregator: 'CacheStatusAggregator'):
        self.aggregator = aggregator
        self.charts = {}
        
    def create_dashboard(self, output_dir: str = "dashboard"):
        """创建仪表板"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib不可用，跳过图表生成，仅生成HTML仪表板")
            # 生成简化的HTML仪表板（不包含图表）
            self._generate_simple_html_dashboard(os.path.join(output_dir, "index.html"))
            logger.info(f"简化仪表板已生成到: {output_dir}")
            return
        
        # 生成各种图表
        self._generate_status_overview_chart(os.path.join(output_dir, "status_overview.png"))
        self._generate_performance_charts(os.path.join(output_dir, "performance.png"))
        self._generate_memory_usage_chart(os.path.join(output_dir, "memory_usage.png"))
        self._generate_hit_rate_trend_chart(os.path.join(output_dir, "hit_rate_trend.png"))
        
        # 生成HTML仪表板
        self._generate_html_dashboard(os.path.join(output_dir, "index.html"))
        
        logger.info(f"仪表板已生成到: {output_dir}")
    
    def _generate_status_overview_chart(self, output_path: str):
        """生成状态概览图表"""
        try:
            current_status = self.aggregator.get_current_status()
            if not current_status:
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('X9缓存状态概览', fontsize=16)
            
            # 缓存状态分布饼图
            status_counts = {
                '健康': current_status.get('healthy_caches', 0),
                '警告': current_status.get('warning_caches', 0),
                '严重': current_status.get('critical_caches', 0)
            }
            
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            ax1.pie(status_counts.values(), labels=status_counts.keys(), colors=colors, autopct='%1.1f%%')
            ax1.set_title('缓存状态分布')
            
            # 内存使用情况
            memory_data = [
                current_status.get('total_memory_usage', 0) / 1024 / 1024,
                current_status.get('total_max_memory', 0) / 1024 / 1024
            ]
            ax2.bar(['已使用', '总容量'], memory_data, color=['#3498db', '#95a5a6'])
            ax2.set_title('内存使用情况 (MB)')
            ax2.set_ylabel('内存 (MB)')
            
            # 命中率
            hit_rate = current_status.get('avg_hit_rate', 0) * 100
            ax3.bar(['命中率'], [hit_rate], color='#2ecc71')
            ax3.set_title('平均命中率')
            ax3.set_ylabel('百分比 (%)')
            ax3.set_ylim(0, 100)
            
            # 整体健康分数
            health_score = current_status.get('overall_health_score', 0)
            ax4.bar(['健康分数'], [health_score], color='#9b59b6')
            ax4.set_title('整体健康分数')
            ax4.set_ylabel('分数')
            ax4.set_ylim(0, 100)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"生成状态概览图表时出错: {e}")
    
    def _generate_performance_charts(self, output_path: str):
        """生成性能图表"""
        try:
            history = self.aggregator.get_history(hours=24)
            if len(history) < 2:
                return
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle('X9缓存性能趋势', fontsize=16)
            
            # 提取时间序列数据
            timestamps = [datetime.fromisoformat(record['timestamp']) for record in history]
            hit_rates = [record['avg_hit_rate'] * 100 for record in history]
            response_times = [record['avg_response_time'] for record in history]
            
            # 命中率趋势
            ax1.plot(timestamps, hit_rates, marker='o', linewidth=2, color='#2ecc71')
            ax1.set_title('命中率趋势')
            ax1.set_ylabel('命中率 (%)')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            # 响应时间趋势
            ax2.plot(timestamps, response_times, marker='s', linewidth=2, color='#e74c3c')
            ax2.set_title('响应时间趋势')
            ax2.set_ylabel('响应时间 (ms)')
            ax2.set_xlabel('时间')
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"生成性能图表时出错: {e}")
    
    def _generate_memory_usage_chart(self, output_path: str):
        """生成内存使用图表"""
        try:
            current_status = self.aggregator.get_current_status()
            if not current_status:
                return
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # 内存使用分布
            used = current_status.get('total_memory_usage', 0) / 1024 / 1024
            total = current_status.get('total_max_memory', 0) / 1024 / 1024
            free = total - used
            
            sizes = [used, free]
            labels = [f'已使用\n{used:.1f} MB', f'空闲\n{free:.1f} MB']
            colors = ['#3498db', '#ecf0f1']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('内存使用分布')
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"生成内存使用图表时出错: {e}")
    
    def _generate_hit_rate_trend_chart(self, output_path: str):
        """生成命中率趋势图表"""
        try:
            history = self.aggregator.get_history(hours=24)
            if len(history) < 2:
                return
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            # 按小时聚合数据
            hourly_data = defaultdict(list)
            for record in history:
                timestamp = datetime.fromisoformat(record['timestamp'])
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
                hourly_data[hour_key].append(record['avg_hit_rate'] * 100)
            
            hours = sorted(hourly_data.keys())
            avg_hit_rates = [statistics.mean(hourly_data[hour]) for hour in hours]
            
            ax.plot(hours, avg_hit_rates, marker='o', linewidth=3, markersize=8, color='#2ecc71')
            ax.fill_between(hours, avg_hit_rates, alpha=0.3, color='#2ecc71')
            ax.set_title('24小时命中率趋势', fontsize=14)
            ax.set_ylabel('命中率 (%)')
            ax.set_xlabel('时间')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            # 添加平均线
            overall_avg = statistics.mean(avg_hit_rates)
            ax.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.7, label=f'平均值: {overall_avg:.1f}%')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"生成命中率趋势图表时出错: {e}")
    
    def _generate_html_dashboard(self, output_path: str):
        """生成HTML仪表板"""
        html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>X9缓存状态仪表板</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .chart-container h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .chart-container img {
            width: 100%;
            height: auto;
            border-radius: 5px;
        }
        .status-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .status-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status-card h4 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .status-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .healthy { color: #2ecc71; }
        .warning { color: #f39c12; }
        .critical { color: #e74c3c; }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>X9缓存状态仪表板</h1>
            <p>实时监控缓存系统状态和性能</p>
            <p>最后更新: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
        
        <div class="status-summary">
            <div class="status-card">
                <h4>总缓存数</h4>
                <div class="value" id="total-caches">-</div>
            </div>
            <div class="status-card">
                <h4>健康缓存</h4>
                <div class="value healthy" id="healthy-caches">-</div>
            </div>
            <div class="status-card">
                <h4>警告缓存</h4>
                <div class="value warning" id="warning-caches">-</div>
            </div>
            <div class="status-card">
                <h4>严重缓存</h4>
                <div class="value critical" id="critical-caches">-</div>
            </div>
            <div class="status-card">
                <h4>整体健康分数</h4>
                <div class="value" id="health-score">-</div>
            </div>
        </div>
        
        <div class="dashboard-grid">
            <div class="chart-container">
                <h3>状态概览</h3>
                <img src="status_overview.png" alt="状态概览">
            </div>
            
            <div class="chart-container">
                <h3>性能趋势</h3>
                <img src="performance.png" alt="性能趋势">
            </div>
            
            <div class="chart-container">
                <h3>内存使用</h3>
                <img src="memory_usage.png" alt="内存使用">
            </div>
            
            <div class="chart-container">
                <h3>命中率趋势</h3>
                <img src="hit_rate_trend.png" alt="命中率趋势">
            </div>
        </div>
        
        <div class="footer">
            <p>© 2025 X9缓存状态聚合器 - 提供专业的缓存监控解决方案</p>
        </div>
    </div>
    
    <script>
        // 自动刷新页面
        setTimeout(function() {
            window.location.reload();
        }, 300000); // 5分钟刷新一次
    </script>
</body>
</html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_simple_html_dashboard(self, output_path: str):
        """生成简化的HTML仪表板（无图表）"""
        html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>X9缓存状态仪表板</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        .status-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .status-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status-card h4 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .status-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .notice {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
            color: #856404;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>X9缓存状态仪表板</h1>
            <p>实时监控缓存系统状态和性能</p>
            <p>最后更新: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
        
        <div class="notice">
            <strong>注意:</strong> 图表功能需要安装matplotlib和pandas库。当前显示简化版本。
        </div>
        
        <div class="status-summary">
            <div class="status-card">
                <h4>总缓存数</h4>
                <div class="value" id="total-caches">-</div>
            </div>
            <div class="status-card">
                <h4>健康缓存</h4>
                <div class="value" id="healthy-caches">-</div>
            </div>
            <div class="status-card">
                <h4>警告缓存</h4>
                <div class="value" id="warning-caches">-</div>
            </div>
            <div class="status-card">
                <h4>严重缓存</h4>
                <div class="value" id="critical-caches">-</div>
            </div>
            <div class="status-card">
                <h4>整体健康分数</h4>
                <div class="value" id="health-score">-</div>
            </div>
        </div>
        
        <div class="footer">
            <p>© 2025 X9缓存状态聚合器 - 提供专业的缓存监控解决方案</p>
            <p>如需完整图表功能，请安装: pip install matplotlib pandas</p>
        </div>
    </div>
    
    <script>
        // 自动刷新页面
        setTimeout(function() {
            window.location.reload();
        }, 300000); // 5分钟刷新一次
    </script>
</body>
</html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


class CacheStatusAggregator:
    """缓存状态聚合器主类"""
    
    def __init__(self, db_path: str = "cache_status_history.db"):
        self.status_collector = StatusCollector()
        self.data_aggregator = DataAggregator()
        self.status_analyzer = StatusAnalyzer()
        self.alert_manager = AlertManager()
        self.history_manager = HistoryManager(db_path)
        self.report_generator = ReportGenerator()
        self.dashboard = Dashboard(self)
        
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 30  # 30秒监控间隔
        
        # 默认预警规则
        self._setup_default_alert_rules()
        
        logger.info("缓存状态聚合器初始化完成")
    
    def _setup_default_alert_rules(self):
        """设置默认预警规则"""
        default_rules = [
            AlertRule("低命中率", "hit_rate", 0.8, "<", "medium"),
            AlertRule("高内存使用", "memory_utilization", 0.9, ">", "high"),
            AlertRule("高响应时间", "response_time", 100.0, ">", "medium"),
            AlertRule("错误过多", "error_count", 10, ">", "high"),
            AlertRule("极低命中率", "hit_rate", 0.5, "<", "critical")
        ]
        
        for rule in default_rules:
            self.alert_manager.add_rule(rule)
    
    def register_cache_module(self, name: str, module: Any, collector_func: Callable):
        """注册缓存模块"""
        self.status_collector.register_cache_module(name, module, collector_func)
        logger.info(f"注册缓存模块: {name}")
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring_active:
            logger.warning("监控已在运行中")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("开始缓存状态监控")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("停止缓存状态监控")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集所有缓存状态
                status_list = self.status_collector.collect_all_status()
                
                # 保存历史记录
                for status in status_list:
                    self.history_manager.save_status(status)
                
                # 检查预警
                for status in status_list:
                    new_alerts = self.alert_manager.check_alerts(status)
                    for alert in new_alerts:
                        # 触发预警回调
                        for callback in self.alert_manager.alert_callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                logger.error(f"预警回调执行失败: {e}")
                
                # 清理旧记录（每小时清理一次）
                if int(time.time()) % 3600 == 0:
                    self.history_manager.cleanup_old_records()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_current_status(self) -> Optional[Dict[str, Any]]:
        """获取当前状态"""
        status_list = self.status_collector.collect_all_status()
        if not status_list:
            return None
        
        return self.data_aggregator.aggregate_status(status_list)
    
    def get_cache_status(self, cache_name: str) -> Optional[CacheStatus]:
        """获取指定缓存状态"""
        return self.status_collector.collect_status(cache_name)
    
    def get_history(self, hours: int = 24, cache_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取历史记录"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        return self.history_manager.get_history(
            cache_name=cache_name,
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
    
    def analyze_trends(self, hours: int = 24) -> Dict[str, Any]:
        """分析趋势"""
        history = self.get_history(hours=hours)
        
        # 如果没有历史数据，返回空结果
        if not history:
            return {
                'hit_rate_trend': 'stable',
                'memory_trend': 'stable',
                'response_time_trend': 'stable',
                'overall_trend': 'stable',
                'trends': {},
                'anomalies': []
            }
        
        # 按时间戳聚合历史数据
        aggregated_data = self._aggregate_history_data(history)
        return self.status_analyzer.analyze_trends(aggregated_data)
    
    def _aggregate_history_data(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """聚合历史数据为趋势分析所需的格式"""
        if not history:
            return []
        
        # 按时间戳分组并聚合
        time_groups = {}
        for record in history:
            timestamp = record['timestamp']
            if timestamp not in time_groups:
                time_groups[timestamp] = []
            time_groups[timestamp].append(record)
        
        # 为每个时间点创建聚合记录
        aggregated_history = []
        for timestamp, records in time_groups.items():
            # 计算该时间点的聚合指标
            hit_rates = [r['hit_rate'] for r in records]
            memory_usage = sum(r['memory_usage'] for r in records)
            response_times = [r['avg_response_time'] for r in records]
            error_counts = [r['error_count'] for r in records]
            
            aggregated_record = {
                'timestamp': timestamp,
                'avg_hit_rate': statistics.mean(hit_rates) if hit_rates else 0.0,
                'total_memory_usage': memory_usage,
                'avg_response_time': statistics.mean(response_times) if response_times else 0.0,
                'total_errors': sum(error_counts),
                'cache_count': len(records)
            }
            aggregated_history.append(aggregated_record)
        
        # 按时间戳排序
        aggregated_history.sort(key=lambda x: x['timestamp'])
        return aggregated_history
    
    def detect_anomalies(self, cache_name: str) -> List[str]:
        """检测异常"""
        current_status = self.get_cache_status(cache_name)
        if not current_status:
            return []
        
        history = self.get_history(hours=24, cache_name=cache_name)
        return self.status_analyzer.detect_anomalies(current_status, history)
    
    def generate_report(self, report_type: str = 'summary') -> str:
        """生成报告"""
        current_status = self.get_current_status()
        if not current_status:
            return "无法生成报告：没有可用的缓存状态数据"
        
        # 准备报告数据
        if report_type == 'summary':
            data = current_status
        elif report_type == 'detailed':
            cache_details = [status.to_dict() for status in self.status_collector.collect_all_status()]
            data = {
                'aggregated_status': current_status,
                'cache_details': cache_details
            }
        elif report_type == 'trend':
            trends = self.analyze_trends()
            trend_data = self.data_aggregator.aggregate_by_time_range(
                self.get_history(hours=24), 'hour'
            )
            data = {
                'trends': trends,
                'trend_data': trend_data
            }
        elif report_type == 'alert':
            data = {
                'active_alerts': [alert.to_dict() for alert in self.alert_manager.active_alerts.values()],
                'alert_history': [alert.to_dict() for alert in self.alert_manager.alert_history[-20:]]
            }
        else:
            data = current_status
        
        return self.report_generator.generate_report(report_type, data)
    
    def create_dashboard(self, output_dir: str = "dashboard"):
        """创建仪表板"""
        self.dashboard.create_dashboard(output_dir)
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃预警"""
        return list(self.alert_manager.active_alerts.values())
    
    def add_custom_alert_rule(self, rule_name: str, metric: str, threshold: float, 
                            operator: str, severity: str):
        """添加自定义预警规则"""
        rule = AlertRule(rule_name, metric, threshold, operator, severity)
        self.alert_manager.add_rule(rule)
    
    def remove_alert_rule(self, rule_name: str):
        """移除预警规则"""
        self.alert_manager.remove_rule(rule_name)
    
    def add_alert_callback(self, callback: Callable):
        """添加预警回调函数"""
        self.alert_manager.add_alert_callback(callback)
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'version': '1.0.0',
            'monitoring_active': self.monitoring_active,
            'registered_caches': list(self.status_collector.cache_modules.keys()),
            'alert_rules_count': len(self.alert_manager.alert_rules),
            'active_alerts_count': len(self.alert_manager.active_alerts),
            'database_path': self.history_manager.db_path
        }


# 示例缓存模块和收集函数
class ExampleCacheModule:
    """示例缓存模块"""
    
    def __init__(self, name: str):
        self.name = name
        self.hit_count = 0
        self.miss_count = 0
        self.memory_usage = 1024 * 1024  # 1MB
        self.max_memory = 10 * 1024 * 1024  # 10MB
        self.entry_count = 100
        self.max_entries = 1000
        self.response_time = 50.0  # ms
        self.error_count = 0
        self.last_access = datetime.now()
    
    def simulate_usage(self):
        """模拟缓存使用"""
        import random
        
        # 模拟命中率波动
        if random.random() < 0.85:  # 85% 命中率
            self.hit_count += 1
        else:
            self.miss_count += 1
        
        # 模拟内存使用增长
        if random.random() < 0.3:  # 30% 概率增长
            self.memory_usage = min(self.memory_usage + random.randint(1024, 10240), self.max_memory)
        
        # 模拟响应时间变化
        self.response_time = max(10.0, self.response_time + random.uniform(-5, 5))
        
        # 模拟偶尔的错误
        if random.random() < 0.02:  # 2% 错误率
            self.error_count += 1
        
        self.last_access = datetime.now()
    
    def get_status_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        miss_rate = self.miss_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'status': 'healthy' if hit_rate > 0.8 else 'warning' if hit_rate > 0.6 else 'critical',
            'hit_rate': hit_rate,
            'miss_rate': miss_rate,
            'memory_usage': self.memory_usage,
            'max_memory': self.max_memory,
            'entry_count': self.entry_count,
            'max_entries': self.max_entries,
            'avg_response_time': self.response_time,
            'last_access_time': self.last_access.isoformat(),
            'error_count': self.error_count
        }


def example_collector(cache_module: ExampleCacheModule) -> Dict[str, Any]:
    """示例收集函数"""
    cache_module.simulate_usage()
    return cache_module.get_status_dict()


if __name__ == "__main__":
    # 使用示例
    aggregator = CacheStatusAggregator()
    
    # 注册示例缓存模块
    cache1 = ExampleCacheModule("用户缓存")
    cache2 = ExampleCacheModule("商品缓存")
    cache3 = ExampleCacheModule("订单缓存")
    
    aggregator.register_cache_module("user_cache", cache1, example_collector)
    aggregator.register_cache_module("product_cache", cache2, example_collector)
    aggregator.register_cache_module("order_cache", cache3, example_collector)
    
    # 添加预警回调
    def alert_callback(alert: Alert):
        print(f"🚨 预警: {alert.message}")
    
    aggregator.add_alert_callback(alert_callback)
    
    # 开始监控
    aggregator.start_monitoring()
    
    try:
        print("X9缓存状态聚合器运行中...")
        print("按 Ctrl+C 停止")
        
        # 等待一段时间
        time.sleep(60)
        
        # 生成报告
        print("\n=== 概要报告 ===")
        print(aggregator.generate_report('summary'))
        
        print("\n=== 趋势分析 ===")
        trends = aggregator.analyze_trends()
        print(f"趋势分析: {trends}")
        
        print("\n=== 活跃预警 ===")
        alerts = aggregator.get_active_alerts()
        for alert in alerts:
            print(f"- {alert.message}")
        
        # 创建仪表板
        aggregator.create_dashboard("dashboard")
        print("\n仪表板已生成到 dashboard 目录")
        
    except KeyboardInterrupt:
        print("\n正在停止监控...")
    finally:
        aggregator.stop_monitoring()
        print("X9缓存状态聚合器已停止")