"""
I9接口状态聚合器模块

该模块实现了一个完整的接口状态聚合器，用于监控和管理系统中的各种接口状态。
主要功能包括：
- 所有接口状态收集
- 接口健康度评估
- 性能指标聚合
- 状态变更监控
- 告警规则管理
- 状态可视化
- 历史数据分析
- 预测性维护
- 状态报告生成

Author: System
Date: 2025-11-05
"""

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import logging
import statistics
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import weakref
import hashlib


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterfaceStatus(Enum):
    """接口状态枚举"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """健康状态枚举"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class InterfaceMetrics:
    """接口性能指标"""
    interface_id: str
    timestamp: datetime
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    availability: float = 100.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_latency: float = 0.0
    success_rate: float = 100.0
    request_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class InterfaceInfo:
    """接口信息"""
    interface_id: str
    name: str
    url: str
    method: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_check: Optional[datetime] = None
    status: InterfaceStatus = InterfaceStatus.UNKNOWN
    health_score: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_check'] = self.last_check.isoformat() if self.last_check else None
        data['status'] = self.status.value
        return data


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    interface_id: str
    metric: str
    condition: str  # >, <, >=, <=, ==, !=
    threshold: float
    level: AlertLevel
    enabled: bool = True
    cooldown: int = 300  # 秒
    last_triggered: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def check_condition(self, value: float) -> bool:
        """检查是否满足告警条件"""
        try:
            if self.condition == ">":
                return value > self.threshold
            elif self.condition == "<":
                return value < self.threshold
            elif self.condition == ">=":
                return value >= self.threshold
            elif self.condition == "<=":
                return value <= self.threshold
            elif self.condition == "==":
                return abs(value - self.threshold) < 1e-6
            elif self.condition == "!=":
                return abs(value - self.threshold) >= 1e-6
            else:
                logger.warning(f"未知的告警条件: {self.condition}")
                return False
        except Exception as e:
            logger.error(f"检查告警条件时出错: {e}")
            return False


@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    rule_id: str
    interface_id: str
    level: AlertLevel
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['resolved_at'] = self.resolved_at.isoformat() if self.resolved_at else None
        data['level'] = self.level.value
        return data


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str = "interface_state.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建接口信息表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS interfaces (
                        interface_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        url TEXT NOT NULL,
                        method TEXT NOT NULL,
                        description TEXT,
                        tags TEXT,
                        created_at TEXT NOT NULL,
                        last_check TEXT,
                        status TEXT NOT NULL,
                        health_score REAL DEFAULT 100.0
                    )
                """)
                
                # 创建接口指标表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        interface_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        response_time REAL,
                        throughput REAL,
                        error_rate REAL,
                        availability REAL,
                        cpu_usage REAL,
                        memory_usage REAL,
                        network_latency REAL,
                        success_rate REAL,
                        request_count INTEGER,
                        error_count INTEGER,
                        FOREIGN KEY (interface_id) REFERENCES interfaces (interface_id)
                    )
                """)
                
                # 创建告警规则表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alert_rules (
                        rule_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        interface_id TEXT NOT NULL,
                        metric TEXT NOT NULL,
                        condition TEXT NOT NULL,
                        threshold REAL NOT NULL,
                        level TEXT NOT NULL,
                        enabled BOOLEAN DEFAULT 1,
                        cooldown INTEGER DEFAULT 300,
                        last_triggered TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                
                # 创建告警表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        alert_id TEXT PRIMARY KEY,
                        rule_id TEXT NOT NULL,
                        interface_id TEXT NOT NULL,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        threshold REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        resolved BOOLEAN DEFAULT 0,
                        resolved_at TEXT
                    )
                """)
                
                # 创建历史数据分析表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS historical_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        interface_id TEXT NOT NULL,
                        analysis_date TEXT NOT NULL,
                        avg_response_time REAL,
                        avg_throughput REAL,
                        avg_error_rate REAL,
                        availability_percentage REAL,
                        health_score REAL,
                        trend_analysis TEXT,
                        predictions TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                
                conn.commit()
                logger.info("数据库初始化完成")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def execute_query(self, query: str, params: Tuple = ()) -> List[Dict]:
        """执行查询"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"执行查询失败: {e}")
            return []
    
    def execute_update(self, query: str, params: Tuple = ()) -> bool:
        """执行更新"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"执行更新失败: {e}")
            return False


class HealthCalculator:
    """健康度计算器"""
    
    @staticmethod
    def calculate_health_score(metrics: InterfaceMetrics) -> float:
        """计算接口健康度分数 (0-100)"""
        try:
            # 响应时间权重 (30%)
            response_time_score = max(0, 100 - (metrics.response_time * 2))
            
            # 可用性权重 (25%)
            availability_score = metrics.availability
            
            # 成功率权重 (20%)
            success_score = metrics.success_rate
            
            # 错误率权重 (15%)
            error_score = max(0, 100 - (metrics.error_rate * 100))
            
            # 吞吐量权重 (10%)
            throughput_score = min(100, metrics.throughput / 10)
            
            # 综合计算
            total_score = (
                response_time_score * 0.30 +
                availability_score * 0.25 +
                success_score * 0.20 +
                error_score * 0.15 +
                throughput_score * 0.10
            )
            
            return round(total_score, 2)
        except Exception as e:
            logger.error(f"计算健康度分数失败: {e}")
            return 0.0
    
    @staticmethod
    def get_health_status(score: float) -> HealthStatus:
        """根据分数获取健康状态"""
        if score >= 90:
            return HealthStatus.EXCELLENT
        elif score >= 75:
            return HealthStatus.GOOD
        elif score >= 60:
            return HealthStatus.FAIR
        elif score >= 40:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL


class AlertManager:
    """告警管理器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.alert_callbacks: List[Callable] = []
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def create_alert_rule(self, rule: AlertRule) -> bool:
        """创建告警规则"""
        try:
            query = """
                INSERT OR REPLACE INTO alert_rules 
                (rule_id, name, interface_id, metric, condition, threshold, 
                 level, enabled, cooldown, last_triggered, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                rule.rule_id, rule.name, rule.interface_id, rule.metric,
                rule.condition, rule.threshold, rule.level.value,
                rule.enabled, rule.cooldown,
                rule.last_triggered.isoformat() if rule.last_triggered else None,
                rule.created_at.isoformat()
            )
            return self.db_manager.execute_update(query, params)
        except Exception as e:
            logger.error(f"创建告警规则失败: {e}")
            return False
    
    def check_alerts(self, metrics: InterfaceMetrics) -> List[Alert]:
        """检查告警"""
        alerts = []
        try:
            # 获取接口的所有告警规则
            query = """
                SELECT * FROM alert_rules 
                WHERE interface_id = ? AND enabled = 1
            """
            rules_data = self.db_manager.execute_query(query, (metrics.interface_id,))
            
            for rule_data in rules_data:
                rule = AlertRule(
                    rule_id=rule_data['rule_id'],
                    name=rule_data['name'],
                    interface_id=rule_data['interface_id'],
                    metric=rule_data['metric'],
                    condition=rule_data['condition'],
                    threshold=rule_data['threshold'],
                    level=AlertLevel(rule_data['level']),
                    enabled=bool(rule_data['enabled']),
                    cooldown=rule_data['cooldown'],
                    last_triggered=datetime.fromisoformat(rule_data['last_triggered']) 
                        if rule_data['last_triggered'] else None,
                    created_at=datetime.fromisoformat(rule_data['created_at'])
                )
                
                # 获取指标值
                metric_value = getattr(metrics, rule.metric, None)
                if metric_value is not None and rule.check_condition(metric_value):
                    # 检查冷却时间
                    if rule.last_triggered is None or \
                       (datetime.now() - rule.last_triggered).seconds >= rule.cooldown:
                        
                        alert = Alert(
                            alert_id=self._generate_alert_id(),
                            rule_id=rule.rule_id,
                            interface_id=rule.interface_id,
                            level=rule.level,
                            message=f"{rule.name}: {rule.metric} {rule.condition} {rule.threshold}, 当前值: {metric_value}",
                            metric_value=metric_value,
                            threshold=rule.threshold
                        )
                        alerts.append(alert)
                        
                        # 更新规则的最后触发时间
                        rule.last_triggered = datetime.now()
                        self.create_alert_rule(rule)
            
            # 保存告警
            for alert in alerts:
                self.save_alert(alert)
                
                # 触发回调
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"告警回调执行失败: {e}")
            
            return alerts
        except Exception as e:
            logger.error(f"检查告警失败: {e}")
            return []
    
    def save_alert(self, alert: Alert) -> bool:
        """保存告警"""
        try:
            query = """
                INSERT INTO alerts 
                (alert_id, rule_id, interface_id, level, message, metric_value, 
                 threshold, timestamp, resolved, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                alert.alert_id, alert.rule_id, alert.interface_id,
                alert.level.value, alert.message, alert.metric_value,
                alert.threshold, alert.timestamp.isoformat(),
                alert.resolved, alert.resolved_at.isoformat() if alert.resolved_at else None
            )
            return self.db_manager.execute_update(query, params)
        except Exception as e:
            logger.error(f"保存告警失败: {e}")
            return False
    
    def _generate_alert_id(self) -> str:
        """生成告警ID"""
        timestamp = str(int(time.time() * 1000))
        random_str = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"alert_{timestamp}_{random_str}"


class StateMonitor:
    """状态监控器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.state_callbacks: List[Callable] = []
        self.last_states: Dict[str, InterfaceStatus] = {}
    
    def add_state_callback(self, callback: Callable[[str, InterfaceStatus, InterfaceStatus], None]):
        """添加状态变更回调函数"""
        self.state_callbacks.append(callback)
    
    def update_interface_status(self, interface_id: str, status: InterfaceStatus, 
                              health_score: float) -> bool:
        """更新接口状态"""
        try:
            # 检查状态是否变更
            old_status = self.last_states.get(interface_id)
            
            # 更新数据库
            query = """
                UPDATE interfaces 
                SET status = ?, health_score = ?, last_check = ?
                WHERE interface_id = ?
            """
            params = (
                status.value, health_score, datetime.now().isoformat(), interface_id
            )
            success = self.db_manager.execute_update(query, params)
            
            if success:
                # 更新内存中的状态
                self.last_states[interface_id] = status
                
                # 如果状态发生变更，触发回调
                if old_status and old_status != status:
                    for callback in self.state_callbacks:
                        try:
                            callback(interface_id, old_status, status)
                        except Exception as e:
                            logger.error(f"状态变更回调执行失败: {e}")
            
            return success
        except Exception as e:
            logger.error(f"更新接口状态失败: {e}")
            return False
    
    def get_interface_status(self, interface_id: str) -> Optional[InterfaceStatus]:
        """获取接口状态"""
        try:
            query = "SELECT status FROM interfaces WHERE interface_id = ?"
            result = self.db_manager.execute_query(query, (interface_id,))
            if result:
                return InterfaceStatus(result[0]['status'])
            return None
        except Exception as e:
            logger.error(f"获取接口状态失败: {e}")
            return None


class PerformanceAggregator:
    """性能指标聚合器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def aggregate_metrics(self, interface_id: str, time_range: timedelta) -> Dict[str, float]:
        """聚合指定时间范围内的性能指标"""
        try:
            start_time = datetime.now() - time_range
            query = """
                SELECT 
                    AVG(response_time) as avg_response_time,
                    MAX(response_time) as max_response_time,
                    MIN(response_time) as min_response_time,
                    AVG(throughput) as avg_throughput,
                    AVG(error_rate) as avg_error_rate,
                    AVG(availability) as avg_availability,
                    AVG(cpu_usage) as avg_cpu_usage,
                    AVG(memory_usage) as avg_memory_usage,
                    AVG(network_latency) as avg_network_latency,
                    AVG(success_rate) as avg_success_rate,
                    SUM(request_count) as total_requests,
                    SUM(error_count) as total_errors
                FROM metrics 
                WHERE interface_id = ? AND timestamp >= ?
            """
            result = self.db_manager.execute_query(query, (interface_id, start_time.isoformat()))
            
            if result and result[0]:
                row = result[0]
                # 计算错误率
                total_requests = row['total_requests'] or 0
                total_errors = row['total_errors'] or 0
                calculated_error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
                
                return {
                    'avg_response_time': row['avg_response_time'] or 0.0,
                    'max_response_time': row['max_response_time'] or 0.0,
                    'min_response_time': row['min_response_time'] or 0.0,
                    'avg_throughput': row['avg_throughput'] or 0.0,
                    'avg_error_rate': calculated_error_rate,
                    'avg_availability': row['avg_availability'] or 0.0,
                    'avg_cpu_usage': row['avg_cpu_usage'] or 0.0,
                    'avg_memory_usage': row['avg_memory_usage'] or 0.0,
                    'avg_network_latency': row['avg_network_latency'] or 0.0,
                    'avg_success_rate': row['avg_success_rate'] or 0.0,
                    'total_requests': total_requests,
                    'total_errors': total_errors
                }
            return {}
        except Exception as e:
            logger.error(f"聚合性能指标失败: {e}")
            return {}
    
    def get_trend_analysis(self, interface_id: str, days: int = 7) -> Dict[str, Any]:
        """获取趋势分析"""
        try:
            start_time = datetime.now() - timedelta(days=days)
            query = """
                SELECT 
                    DATE(timestamp) as date,
                    AVG(response_time) as avg_response_time,
                    AVG(throughput) as avg_throughput,
                    AVG(error_rate) as avg_error_rate,
                    AVG(availability) as avg_availability,
                    AVG(health_score) as avg_health_score
                FROM (
                    SELECT m.interface_id, m.timestamp, m.response_time, m.throughput,
                           m.error_rate, m.availability, m.cpu_usage, m.memory_usage,
                           m.network_latency, m.success_rate, m.request_count, m.error_count,
                           i.health_score
                    FROM metrics m
                    JOIN interfaces i ON m.interface_id = i.interface_id
                    WHERE m.interface_id = ? AND m.timestamp >= ?
                )
                GROUP BY DATE(timestamp)
                ORDER BY date
            """
            result = self.db_manager.execute_query(query, (interface_id, start_time.isoformat()))
            
            if result:
                dates = [row['date'] for row in result]
                response_times = [row['avg_response_time'] or 0 for row in result]
                throughputs = [row['avg_throughput'] or 0 for row in result]
                error_rates = [row['avg_error_rate'] or 0 for row in result]
                availabilities = [row['avg_availability'] or 0 for row in result]
                health_scores = [row['avg_health_score'] or 0 for row in result]
                
                # 计算趋势
                trend_analysis = {
                    'response_time_trend': self._calculate_trend(response_times),
                    'throughput_trend': self._calculate_trend(throughputs),
                    'error_rate_trend': self._calculate_trend(error_rates),
                    'availability_trend': self._calculate_trend(availabilities),
                    'health_score_trend': self._calculate_trend(health_scores)
                }
                
                return {
                    'dates': dates,
                    'metrics': {
                        'response_time': response_times,
                        'throughput': throughputs,
                        'error_rate': error_rates,
                        'availability': availabilities,
                        'health_score': health_scores
                    },
                    'trend_analysis': trend_analysis
                }
            return {}
        except Exception as e:
            logger.error(f"获取趋势分析失败: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势方向"""
        if len(values) < 2:
            return "stable"
        
        # 计算线性回归斜率
        x = list(range(len(values)))
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"


class PredictiveAnalyzer:
    """预测性分析器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def predict_failure_risk(self, interface_id: str, days_ahead: int = 7) -> Dict[str, Any]:
        """预测故障风险"""
        try:
            # 获取历史数据
            historical_data = self._get_historical_data(interface_id, 30)  # 30天历史数据
            
            if not historical_data:
                return {'risk_level': 'unknown', 'prediction': 'insufficient_data'}
            
            # 分析趋势和模式
            risk_factors = self._analyze_risk_factors(historical_data)
            
            # 预测未来状态
            predictions = self._predict_future_state(historical_data, days_ahead)
            
            # 计算综合风险等级
            risk_level = self._calculate_risk_level(risk_factors, predictions)
            
            return {
                'interface_id': interface_id,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'predictions': predictions,
                'confidence': self._calculate_confidence(historical_data),
                'recommendations': self._generate_recommendations(risk_factors, predictions)
            }
        except Exception as e:
            logger.error(f"预测故障风险失败: {e}")
            return {'risk_level': 'error', 'error': str(e)}
    
    def _get_historical_data(self, interface_id: str, days: int) -> List[Dict]:
        """获取历史数据"""
        start_time = datetime.now() - timedelta(days=days)
        query = """
            SELECT * FROM metrics 
            WHERE interface_id = ? AND timestamp >= ?
            ORDER BY timestamp
        """
        return self.db_manager.execute_query(query, (interface_id, start_time.isoformat()))
    
    def _analyze_risk_factors(self, historical_data: List[Dict]) -> Dict[str, float]:
        """分析风险因子"""
        if not historical_data:
            return {}
        
        response_times = [row['response_time'] for row in historical_data if row['response_time']]
        error_rates = [row['error_rate'] for row in historical_data if row['error_rate']]
        availabilities = [row['availability'] for row in historical_data if row['availability']]
        
        risk_factors = {}
        
        # 响应时间波动性
        if response_times:
            rt_std = statistics.stdev(response_times) if len(response_times) > 1 else 0
            rt_mean = statistics.mean(response_times)
            risk_factors['response_time_volatility'] = rt_std / rt_mean if rt_mean > 0 else 0
        
        # 错误率趋势
        if error_rates:
            error_trend = self._calculate_trend_slope(error_rates)
            risk_factors['error_rate_trend'] = error_trend
        
        # 可用性下降趋势
        if availabilities:
            availability_trend = self._calculate_trend_slope(availabilities)
            risk_factors['availability_degradation'] = -availability_trend  # 负值表示下降
        
        return risk_factors
    
    def _predict_future_state(self, historical_data: List[Dict], days_ahead: int) -> Dict[str, float]:
        """预测未来状态"""
        if not historical_data:
            return {}
        
        # 简单的线性预测
        recent_data = historical_data[-min(24, len(historical_data)):]  # 最近24个数据点
        
        predictions = {}
        
        # 预测响应时间
        response_times = [row['response_time'] for row in recent_data if row['response_time']]
        if response_times:
            trend = self._calculate_trend_slope(response_times)
            last_value = response_times[-1]
            predictions['predicted_response_time'] = last_value + (trend * days_ahead)
        
        # 预测错误率
        error_rates = [row['error_rate'] for row in recent_data if row['error_rate']]
        if error_rates:
            trend = self._calculate_trend_slope(error_rates)
            last_value = error_rates[-1]
            predictions['predicted_error_rate'] = max(0, last_value + (trend * days_ahead))
        
        # 预测可用性
        availabilities = [row['availability'] for row in recent_data if row['availability']]
        if availabilities:
            trend = self._calculate_trend_slope(availabilities)
            last_value = availabilities[-1]
            predictions['predicted_availability'] = max(0, min(100, last_value + (trend * days_ahead)))
        
        return predictions
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """计算趋势斜率"""
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
        except ZeroDivisionError:
            return 0.0
    
    def _calculate_risk_level(self, risk_factors: Dict[str, float], 
                            predictions: Dict[str, float]) -> str:
        """计算风险等级"""
        risk_score = 0
        
        # 响应时间波动性风险
        if 'response_time_volatility' in risk_factors:
            volatility = risk_factors['response_time_volatility']
            if volatility > 0.5:
                risk_score += 30
            elif volatility > 0.2:
                risk_score += 15
        
        # 错误率趋势风险
        if 'error_rate_trend' in risk_factors:
            trend = risk_factors['error_rate_trend']
            if trend > 0.1:
                risk_score += 40
            elif trend > 0.05:
                risk_score += 20
        
        # 可用性下降风险
        if 'availability_degradation' in risk_factors:
            degradation = risk_factors['availability_degradation']
            if degradation > 0.1:
                risk_score += 30
            elif degradation > 0.05:
                risk_score += 15
        
        # 预测值风险
        if 'predicted_error_rate' in predictions:
            pred_error = predictions['predicted_error_rate']
            if pred_error > 10:
                risk_score += 50
            elif pred_error > 5:
                risk_score += 25
        
        if 'predicted_availability' in predictions:
            pred_avail = predictions['predicted_availability']
            if pred_avail < 90:
                risk_score += 30
            elif pred_avail < 95:
                risk_score += 15
        
        # 确定风险等级
        if risk_score >= 80:
            return 'high'
        elif risk_score >= 50:
            return 'medium'
        elif risk_score >= 20:
            return 'low'
        else:
            return 'minimal'
    
    def _calculate_confidence(self, historical_data: List[Dict]) -> float:
        """计算预测置信度"""
        if not historical_data:
            return 0.0
        
        # 基于数据量和一致性计算置信度
        data_volume_score = min(1.0, len(historical_data) / 100)  # 100个数据点为满分
        
        # 计算数据一致性
        if len(historical_data) > 1:
            response_times = [row['response_time'] for row in historical_data if row['response_time']]
            if response_times:
                cv = statistics.stdev(response_times) / statistics.mean(response_times) if statistics.mean(response_times) > 0 else 1
                consistency_score = max(0, 1 - cv)  # 变异系数越小，置信度越高
            else:
                consistency_score = 0.5
        else:
            consistency_score = 0.5
        
        return (data_volume_score * 0.6 + consistency_score * 0.4) * 100
    
    def _generate_recommendations(self, risk_factors: Dict[str, float], 
                                predictions: Dict[str, float]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于风险因子的建议
        if 'response_time_volatility' in risk_factors:
            volatility = risk_factors['response_time_volatility']
            if volatility > 0.5:
                recommendations.append("响应时间波动较大，建议检查系统负载和资源使用情况")
            elif volatility > 0.2:
                recommendations.append("响应时间存在一定波动，建议持续监控")
        
        if 'error_rate_trend' in risk_factors:
            trend = risk_factors['error_rate_trend']
            if trend > 0.1:
                recommendations.append("错误率呈上升趋势，建议立即检查接口代码和依赖服务")
            elif trend > 0.05:
                recommendations.append("错误率有上升迹象，建议加强监控")
        
        if 'availability_degradation' in risk_factors:
            degradation = risk_factors['availability_degradation']
            if degradation > 0.1:
                recommendations.append("可用性下降明显，建议检查系统健康状况")
            elif degradation > 0.05:
                recommendations.append("可用性有下降趋势，建议定期检查")
        
        # 基于预测的建议
        if 'predicted_error_rate' in predictions:
            pred_error = predictions['predicted_error_rate']
            if pred_error > 10:
                recommendations.append("预测错误率将超过10%，建议立即采取预防措施")
            elif pred_error > 5:
                recommendations.append("预测错误率将超过5%，建议准备应急预案")
        
        if 'predicted_availability' in predictions:
            pred_avail = predictions['predicted_availability']
            if pred_avail < 90:
                recommendations.append("预测可用性将低于90%，建议检查系统稳定性")
            elif pred_avail < 95:
                recommendations.append("预测可用性将低于95%，建议优化系统性能")
        
        if not recommendations:
            recommendations.append("系统状态良好，建议继续保持当前监控策略")
        
        return recommendations


class VisualizationEngine:
    """可视化引擎"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def create_dashboard_data(self, interface_ids: List[str] = None) -> Dict[str, Any]:
        """创建仪表板数据"""
        try:
            if not interface_ids:
                # 获取所有接口
                query = "SELECT interface_id, name, status, health_score FROM interfaces"
                interfaces_data = self.db_manager.execute_query(query)
                interface_ids = [row['interface_id'] for row in interfaces_data]
            
            dashboard_data = {
                'interfaces': [],
                'alerts': [],
                'summary': {},
                'charts': {}
            }
            
            # 获取接口信息
            for interface_id in interface_ids:
                interface_info = self._get_interface_info(interface_id)
                if interface_info:
                    dashboard_data['interfaces'].append(interface_info)
            
            # 获取活跃告警
            active_alerts = self._get_active_alerts()
            dashboard_data['alerts'] = active_alerts
            
            # 生成汇总信息
            dashboard_data['summary'] = self._generate_summary(interface_ids, active_alerts)
            
            # 生成图表数据
            dashboard_data['charts'] = self._generate_chart_data(interface_ids)
            
            return dashboard_data
        except Exception as e:
            logger.error(f"创建仪表板数据失败: {e}")
            return {}
    
    def _get_interface_info(self, interface_id: str) -> Optional[Dict]:
        """获取接口详细信息"""
        try:
            query = """
                SELECT i.*, 
                       AVG(m.response_time) as avg_response_time,
                       AVG(m.throughput) as avg_throughput,
                       AVG(m.error_rate) as avg_error_rate,
                       AVG(m.availability) as avg_availability
                FROM interfaces i
                LEFT JOIN metrics m ON i.interface_id = m.interface_id
                WHERE i.interface_id = ?
                GROUP BY i.interface_id
            """
            result = self.db_manager.execute_query(query, (interface_id,))
            return result[0] if result else None
        except Exception as e:
            logger.error(f"获取接口信息失败: {e}")
            return None
    
    def _get_active_alerts(self) -> List[Dict]:
        """获取活跃告警"""
        try:
            query = """
                SELECT a.*, i.name as interface_name
                FROM alerts a
                JOIN interfaces i ON a.interface_id = i.interface_id
                WHERE a.resolved = 0
                ORDER BY a.timestamp DESC
                LIMIT 50
            """
            return self.db_manager.execute_query(query)
        except Exception as e:
            logger.error(f"获取活跃告警失败: {e}")
            return []
    
    def _generate_summary(self, interface_ids: List[str], alerts: List[Dict]) -> Dict[str, Any]:
        """生成汇总信息"""
        try:
            total_interfaces = len(interface_ids)
            
            # 统计各状态接口数量
            status_counts = defaultdict(int)
            for interface_id in interface_ids:
                status = self.db_manager.execute_query(
                    "SELECT status FROM interfaces WHERE interface_id = ?", (interface_id,)
                )
                if status:
                    status_counts[status[0]['status']] += 1
            
            # 统计告警数量
            alert_counts = defaultdict(int)
            for alert in alerts:
                alert_counts[alert['level']] += 1
            
            # 计算平均健康分数
            health_scores = []
            for interface_id in interface_ids:
                score = self.db_manager.execute_query(
                    "SELECT health_score FROM interfaces WHERE interface_id = ?", (interface_id,)
                )
                if score:
                    health_scores.append(score[0]['health_score'])
            
            avg_health_score = statistics.mean(health_scores) if health_scores else 0
            
            return {
                'total_interfaces': total_interfaces,
                'status_distribution': dict(status_counts),
                'alert_summary': dict(alert_counts),
                'average_health_score': round(avg_health_score, 2),
                'critical_alerts': alert_counts.get('critical', 0),
                'error_alerts': alert_counts.get('error', 0),
                'warning_alerts': alert_counts.get('warning', 0)
            }
        except Exception as e:
            logger.error(f"生成汇总信息失败: {e}")
            return {}
    
    def _generate_chart_data(self, interface_ids: List[str]) -> Dict[str, Any]:
        """生成图表数据"""
        try:
            chart_data = {
                'health_score_trend': [],
                'response_time_trend': [],
                'error_rate_trend': [],
                'availability_trend': []
            }
            
            # 获取最近24小时的数据
            start_time = datetime.now() - timedelta(hours=24)
            
            for interface_id in interface_ids[:5]:  # 只显示前5个接口
                query = """
                    SELECT timestamp, health_score, response_time, error_rate, availability
                    FROM (
                        SELECT m.timestamp, i.health_score, m.response_time, m.error_rate, m.availability
                        FROM metrics m
                        JOIN interfaces i ON m.interface_id = i.interface_id
                        WHERE m.interface_id = ? AND m.timestamp >= ?
                        ORDER BY m.timestamp
                    )
                """
                result = self.db_manager.execute_query(query, (interface_id, start_time.isoformat()))
                
                if result:
                    chart_data['health_score_trend'].append({
                        'interface_id': interface_id,
                        'data': [(row['timestamp'], row['health_score']) for row in result if row['health_score']]
                    })
                    chart_data['response_time_trend'].append({
                        'interface_id': interface_id,
                        'data': [(row['timestamp'], row['response_time']) for row in result if row['response_time']]
                    })
                    chart_data['error_rate_trend'].append({
                        'interface_id': interface_id,
                        'data': [(row['timestamp'], row['error_rate']) for row in result if row['error_rate']]
                    })
                    chart_data['availability_trend'].append({
                        'interface_id': interface_id,
                        'data': [(row['timestamp'], row['availability']) for row in result if row['availability']]
                    })
            
            return chart_data
        except Exception as e:
            logger.error(f"生成图表数据失败: {e}")
            return {}
    
    def generate_performance_chart(self, interface_id: str, days: int = 7) -> str:
        """生成性能图表"""
        try:
            start_time = datetime.now() - timedelta(days=days)
            query = """
                SELECT DATE(timestamp) as date,
                       AVG(response_time) as avg_response_time,
                       AVG(throughput) as avg_throughput,
                       AVG(error_rate) as avg_error_rate,
                       AVG(availability) as avg_availability
                FROM metrics 
                WHERE interface_id = ? AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """
            result = self.db_manager.execute_query(query, (interface_id, start_time.isoformat()))
            
            if not result:
                return ""
            
            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'接口 {interface_id} 性能趋势 ({days}天)', fontsize=16)
            
            dates = [datetime.strptime(row['date'], '%Y-%m-%d') for row in result]
            
            # 响应时间趋势
            response_times = [row['avg_response_time'] or 0 for row in result]
            ax1.plot(dates, response_times, 'b-o', linewidth=2, markersize=6)
            ax1.set_title('平均响应时间')
            ax1.set_ylabel('响应时间 (ms)')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            
            # 吞吐量趋势
            throughputs = [row['avg_throughput'] or 0 for row in result]
            ax2.plot(dates, throughputs, 'g-o', linewidth=2, markersize=6)
            ax2.set_title('平均吞吐量')
            ax2.set_ylabel('吞吐量 (req/s)')
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            
            # 错误率趋势
            error_rates = [row['avg_error_rate'] or 0 for row in result]
            ax3.plot(dates, error_rates, 'r-o', linewidth=2, markersize=6)
            ax3.set_title('平均错误率')
            ax3.set_ylabel('错误率 (%)')
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            
            # 可用性趋势
            availabilities = [row['avg_availability'] or 0 for row in result]
            ax4.plot(dates, availabilities, 'purple', linewidth=2, markersize=6)
            ax4.set_title('平均可用性')
            ax4.set_ylabel('可用性 (%)')
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = f"performance_chart_{interface_id}_{int(time.time())}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
        except Exception as e:
            logger.error(f"生成性能图表失败: {e}")
            return ""


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def generate_status_report(self, interface_ids: List[str] = None, 
                             days: int = 7) -> Dict[str, Any]:
        """生成状态报告"""
        try:
            if not interface_ids:
                interfaces_data = self.db_manager.execute_query(
                    "SELECT interface_id FROM interfaces"
                )
                interface_ids = [row['interface_id'] for row in interfaces_data]
            
            report = {
                'report_info': {
                    'generated_at': datetime.now().isoformat(),
                    'period_days': days,
                    'interface_count': len(interface_ids)
                },
                'executive_summary': {},
                'detailed_analysis': {},
                'alerts_summary': {},
                'recommendations': []
            }
            
            # 执行摘要
            report['executive_summary'] = self._generate_executive_summary(interface_ids, days)
            
            # 详细分析
            report['detailed_analysis'] = self._generate_detailed_analysis(interface_ids, days)
            
            # 告警摘要
            report['alerts_summary'] = self._generate_alerts_summary(interface_ids, days)
            
            # 建议
            report['recommendations'] = self._generate_recommendations(interface_ids, days)
            
            return report
        except Exception as e:
            logger.error(f"生成状态报告失败: {e}")
            return {'error': str(e)}
    
    def _generate_executive_summary(self, interface_ids: List[str], days: int) -> Dict[str, Any]:
        """生成执行摘要"""
        try:
            total_interfaces = len(interface_ids)
            
            # 统计状态分布
            status_distribution = defaultdict(int)
            health_scores = []
            
            for interface_id in interface_ids:
                # 获取接口状态
                status_data = self.db_manager.execute_query(
                    "SELECT status, health_score FROM interfaces WHERE interface_id = ?",
                    (interface_id,)
                )
                if status_data:
                    status_distribution[status_data[0]['status']] += 1
                    health_scores.append(status_data[0]['health_score'])
            
            # 计算关键指标
            avg_health_score = statistics.mean(health_scores) if health_scores else 0
            online_count = status_distribution.get('online', 0)
            offline_count = status_distribution.get('offline', 0)
            degraded_count = status_distribution.get('degraded', 0)
            
            # 计算可用性
            total_possible_time = total_interfaces * days * 24 * 60  # 总分钟数
            uptime_query = """
                SELECT SUM(availability * ? / 100) as total_uptime_minutes
                FROM (
                    SELECT AVG(availability) as availability
                    FROM metrics 
                    WHERE interface_id = ? AND timestamp >= ?
                    GROUP BY interface_id
                )
            """
            uptime_data = self.db_manager.execute_query(uptime_query, (days * 24 * 60, interface_ids[0] if interface_ids else "", (datetime.now() - timedelta(days=days)).isoformat()))
            
            overall_availability = 100.0
            if uptime_data and uptime_data[0]['total_uptime_minutes'] and total_possible_time > 0:
                overall_availability = (uptime_data[0]['total_uptime_minutes'] / total_possible_time) * 100
            
            return {
                'total_interfaces': total_interfaces,
                'online_interfaces': online_count,
                'offline_interfaces': offline_count,
                'degraded_interfaces': degraded_count,
                'average_health_score': round(avg_health_score, 2),
                'overall_availability': round(overall_availability, 2),
                'system_status': self._determine_overall_status(status_distribution, avg_health_score)
            }
        except Exception as e:
            logger.error(f"生成执行摘要失败: {e}")
            return {}
    
    def _generate_detailed_analysis(self, interface_ids: List[str], days: int) -> Dict[str, Any]:
        """生成详细分析"""
        try:
            detailed_analysis = {}
            
            for interface_id in interface_ids:
                # 获取性能聚合数据
                start_time = datetime.now() - timedelta(days=days)
                query = """
                    SELECT 
                        AVG(response_time) as avg_response_time,
                        MAX(response_time) as max_response_time,
                        AVG(throughput) as avg_throughput,
                        AVG(error_rate) as avg_error_rate,
                        AVG(availability) as avg_availability,
                        SUM(request_count) as total_requests,
                        SUM(error_count) as total_errors,
                        AVG(cpu_usage) as avg_cpu_usage,
                        AVG(memory_usage) as avg_memory_usage
                    FROM metrics 
                    WHERE interface_id = ? AND timestamp >= ?
                """
                result = self.db_manager.execute_query(query, (interface_id, start_time.isoformat()))
                
                if result and result[0]:
                    row = result[0]
                    total_requests = row['total_requests'] or 0
                    total_errors = row['total_errors'] or 0
                    calculated_error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
                    
                    detailed_analysis[interface_id] = {
                        'performance_metrics': {
                            'avg_response_time': round(row['avg_response_time'] or 0, 2),
                            'max_response_time': round(row['max_response_time'] or 0, 2),
                            'avg_throughput': round(row['avg_throughput'] or 0, 2),
                            'avg_error_rate': round(calculated_error_rate, 2),
                            'avg_availability': round(row['avg_availability'] or 0, 2),
                            'total_requests': total_requests,
                            'total_errors': total_errors
                        },
                        'resource_usage': {
                            'avg_cpu_usage': round(row['avg_cpu_usage'] or 0, 2),
                            'avg_memory_usage': round(row['avg_memory_usage'] or 0, 2)
                        },
                        'performance_grade': self._grade_performance(row)
                    }
            
            return detailed_analysis
        except Exception as e:
            logger.error(f"生成详细分析失败: {e}")
            return {}
    
    def _generate_alerts_summary(self, interface_ids: List[str], days: int) -> Dict[str, Any]:
        """生成告警摘要"""
        try:
            start_time = datetime.now() - timedelta(days=days)
            
            # 统计告警数量
            alert_query = """
                SELECT level, COUNT(*) as count
                FROM alerts 
                WHERE interface_id IN ({}) AND timestamp >= ?
                GROUP BY level
            """.format(','.join(['?' for _ in interface_ids]))
            
            alert_stats = self.db_manager.execute_query(
                alert_query, (*interface_ids, start_time.isoformat())
            )
            
            # 获取最近的告警
            recent_alerts_query = """
                SELECT a.*, i.name as interface_name
                FROM alerts a
                JOIN interfaces i ON a.interface_id = i.interface_id
                WHERE a.interface_id IN ({}) AND a.timestamp >= ?
                ORDER BY a.timestamp DESC
                LIMIT 10
            """.format(','.join(['?' for _ in interface_ids]))
            
            recent_alerts = self.db_manager.execute_query(
                recent_alerts_query, (*interface_ids, start_time.isoformat())
            )
            
            # 统计告警分布
            alert_distribution = defaultdict(int)
            for alert_stat in alert_stats:
                alert_distribution[alert_stat['level']] = alert_stat['count']
            
            return {
                'alert_summary': dict(alert_distribution),
                'total_alerts': sum(alert_distribution.values()),
                'critical_alerts': alert_distribution.get('critical', 0),
                'error_alerts': alert_distribution.get('error', 0),
                'warning_alerts': alert_distribution.get('warning', 0),
                'info_alerts': alert_distribution.get('info', 0),
                'recent_alerts': recent_alerts
            }
        except Exception as e:
            logger.error(f"生成告警摘要失败: {e}")
            return {}
    
    def _generate_recommendations(self, interface_ids: List[str], days: int) -> List[str]:
        """生成建议"""
        recommendations = []
        
        try:
            # 基于整体系统状态生成建议
            for interface_id in interface_ids:
                # 获取接口状态
                status_data = self.db_manager.execute_query(
                    "SELECT status, health_score FROM interfaces WHERE interface_id = ?",
                    (interface_id,)
                )
                
                if status_data:
                    status = status_data[0]['status']
                    health_score = status_data[0]['health_score']
                    
                    if status == 'offline':
                        recommendations.append(f"接口 {interface_id} 当前离线，建议立即检查网络连接和服务状态")
                    elif status == 'degraded':
                        recommendations.append(f"接口 {interface_id} 性能下降，建议检查系统负载和资源使用情况")
                    elif health_score < 60:
                        recommendations.append(f"接口 {interface_id} 健康分数较低 ({health_score})，建议进行性能优化")
            
            # 基于错误率生成建议
            start_time = datetime.now() - timedelta(days=days)
            high_error_query = """
                SELECT interface_id, AVG(error_rate) as avg_error_rate
                FROM metrics 
                WHERE interface_id IN ({}) AND timestamp >= ?
                GROUP BY interface_id
                HAVING AVG(error_rate) > 5
            """.format(','.join(['?' for _ in interface_ids]))
            
            high_error_interfaces = self.db_manager.execute_query(
                high_error_query, (*interface_ids, start_time.isoformat())
            )
            
            for row in high_error_interfaces:
                recommendations.append(
                    f"接口 {row['interface_id']} 平均错误率较高 ({row['avg_error_rate']:.2f}%)，"
                    f"建议检查错误日志和异常处理逻辑"
                )
            
            if not recommendations:
                recommendations.append("系统整体运行良好，建议继续保持当前的监控和维护策略")
            
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            recommendations.append("生成建议时出现错误，建议手动检查系统状态")
        
        return recommendations
    
    def _determine_overall_status(self, status_distribution: Dict[str, int], 
                                avg_health_score: float) -> str:
        """确定整体系统状态"""
        total = sum(status_distribution.values())
        if total == 0:
            return "unknown"
        
        offline_ratio = status_distribution.get('offline', 0) / total
        degraded_ratio = status_distribution.get('degraded', 0) / total
        
        if offline_ratio > 0.1 or avg_health_score < 40:
            return "critical"
        elif offline_ratio > 0.05 or degraded_ratio > 0.2 or avg_health_score < 70:
            return "warning"
        elif avg_health_score >= 85:
            return "excellent"
        else:
            return "good"
    
    def _grade_performance(self, metrics_row: Dict) -> str:
        """评级性能"""
        try:
            response_time = metrics_row['avg_response_time'] or 0
            error_rate = metrics_row['avg_error_rate'] or 0
            availability = metrics_row['avg_availability'] or 0
            
            score = 0
            
            # 响应时间评分 (40%)
            if response_time < 100:
                score += 40
            elif response_time < 500:
                score += 30
            elif response_time < 1000:
                score += 20
            else:
                score += 10
            
            # 错误率评分 (30%)
            if error_rate < 1:
                score += 30
            elif error_rate < 3:
                score += 20
            elif error_rate < 5:
                score += 10
            else:
                score += 0
            
            # 可用性评分 (30%)
            if availability >= 99.9:
                score += 30
            elif availability >= 99:
                score += 25
            elif availability >= 95:
                score += 15
            else:
                score += 5
            
            # 确定等级
            if score >= 90:
                return "A"
            elif score >= 80:
                return "B"
            elif score >= 70:
                return "C"
            elif score >= 60:
                return "D"
            else:
                return "F"
        except Exception:
            return "Unknown"


class InterfaceStateAggregator:
    """
    接口状态聚合器主类
    
    该类整合了所有接口状态监控功能，提供统一的接口状态管理解决方案。
    主要功能包括：
    - 接口注册与管理
    - 状态监控与收集
    - 性能指标聚合
    - 告警管理
    - 可视化展示
    - 预测性分析
    - 报告生成
    """
    
    def __init__(self, db_path: str = "interface_state.db"):
        """
        初始化接口状态聚合器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_manager = DatabaseManager(db_path)
        self.health_calculator = HealthCalculator()
        self.alert_manager = AlertManager(self.db_manager)
        self.state_monitor = StateMonitor(self.db_manager)
        self.performance_aggregator = PerformanceAggregator(self.db_manager)
        self.predictive_analyzer = PredictiveAnalyzer(self.db_manager)
        self.visualization_engine = VisualizationEngine(self.db_manager)
        self.report_generator = ReportGenerator(self.db_manager)
        
        # 监控相关
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.monitor_interval = 60  # 秒
        self.interface_checkers: Dict[str, Callable] = {}
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # 回调函数
        self.status_change_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # 内存缓存
        self._interface_cache: Dict[str, InterfaceInfo] = {}
        self._metrics_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info("接口状态聚合器初始化完成")
    
    async def start_monitoring(self, interval: int = 60):
        """
        开始监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self.monitoring_active:
            logger.warning("监控已经在运行中")
            return
        
        self.monitor_interval = interval
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"开始接口状态监控，间隔: {interval}秒")
    
    async def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("接口状态监控已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                await self.collect_all_interface_states()
                await asyncio.sleep(self.monitor_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                await asyncio.sleep(5)  # 出错时等待5秒后重试
    
    async def collect_all_interface_states(self):
        """收集所有接口状态"""
        try:
            # 获取所有注册的接口
            interfaces = self.get_all_interfaces()
            
            # 并发检查所有接口
            tasks = []
            for interface in interfaces:
                task = asyncio.create_task(self._check_interface_status(interface))
                tasks.append(task)
            
            # 等待所有任务完成
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"收集接口状态失败: {e}")
    
    async def _check_interface_status(self, interface: InterfaceInfo):
        """检查单个接口状态"""
        try:
            # 获取检查器
            checker = self.interface_checkers.get(interface.interface_id)
            if not checker:
                logger.warning(f"接口 {interface.interface_id} 没有注册检查器")
                return
            
            # 执行检查
            metrics = await checker(interface)
            if metrics:
                # 保存指标
                self.save_metrics(metrics)
                
                # 计算健康分数
                health_score = self.health_calculator.calculate_health_score(metrics)
                
                # 确定状态
                status = self._determine_interface_status(metrics, health_score)
                
                # 更新状态
                self.state_monitor.update_interface_status(
                    interface.interface_id, status, health_score
                )
                
                # 检查告警
                alerts = self.alert_manager.check_alerts(metrics)
                
                # 更新缓存
                self._update_cache(interface.interface_id, metrics, status, health_score)
        
        except Exception as e:
            logger.error(f"检查接口 {interface.interface_id} 状态失败: {e}")
    
    def _determine_interface_status(self, metrics: InterfaceMetrics, 
                                  health_score: float) -> InterfaceStatus:
        """确定接口状态"""
        try:
            # 基于健康分数确定状态
            if health_score >= 90:
                return InterfaceStatus.ONLINE
            elif health_score >= 70:
                return InterfaceStatus.DEGRADED
            elif health_score >= 50:
                return InterfaceStatus.ERROR
            else:
                return InterfaceStatus.OFFLINE
        except Exception as e:
            logger.error(f"确定接口状态失败: {e}")
            return InterfaceStatus.UNKNOWN
    
    def _update_cache(self, interface_id: str, metrics: InterfaceMetrics, 
                     status: InterfaceStatus, health_score: float):
        """更新缓存"""
        try:
            # 更新指标缓存
            self._metrics_cache[interface_id].append(metrics)
            
            # 更新接口信息缓存
            if interface_id in self._interface_cache:
                interface = self._interface_cache[interface_id]
                interface.status = status
                interface.health_score = health_score
                interface.last_check = datetime.now()
        
        except Exception as e:
            logger.error(f"更新缓存失败: {e}")
    
    def register_interface(self, interface: InterfaceInfo, 
                          checker: Callable[[InterfaceInfo], InterfaceMetrics]):
        """
        注册接口
        
        Args:
            interface: 接口信息
            checker: 状态检查函数
        """
        try:
            # 保存到数据库
            query = """
                INSERT OR REPLACE INTO interfaces 
                (interface_id, name, url, method, description, tags, 
                 created_at, last_check, status, health_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                interface.interface_id, interface.name, interface.url,
                interface.method, interface.description, json.dumps(interface.tags),
                interface.created_at.isoformat(), interface.last_check.isoformat() 
                    if interface.last_check else None,
                interface.status.value, interface.health_score
            )
            
            if self.db_manager.execute_update(query, params):
                # 注册检查器
                self.interface_checkers[interface.interface_id] = checker
                
                # 更新缓存
                self._interface_cache[interface.interface_id] = interface
                
                logger.info(f"接口 {interface.interface_id} 注册成功")
                return True
            else:
                logger.error(f"接口 {interface.interface_id} 注册失败")
                return False
        
        except Exception as e:
            logger.error(f"注册接口失败: {e}")
            return False
    
    def unregister_interface(self, interface_id: str) -> bool:
        """注销接口"""
        try:
            # 从数据库删除
            query = "DELETE FROM interfaces WHERE interface_id = ?"
            success = self.db_manager.execute_update(query, (interface_id,))
            
            if success:
                # 移除检查器
                self.interface_checkers.pop(interface_id, None)
                
                # 清除缓存
                self._interface_cache.pop(interface_id, None)
                self._metrics_cache.pop(interface_id, None)
                
                logger.info(f"接口 {interface_id} 注销成功")
            
            return success
        except Exception as e:
            logger.error(f"注销接口失败: {e}")
            return False
    
    def get_interface(self, interface_id: str) -> Optional[InterfaceInfo]:
        """获取接口信息"""
        try:
            # 先从缓存获取
            if interface_id in self._interface_cache:
                return self._interface_cache[interface_id]
            
            # 从数据库获取
            query = "SELECT * FROM interfaces WHERE interface_id = ?"
            result = self.db_manager.execute_query(query, (interface_id,))
            
            if result:
                row = result[0]
                interface = InterfaceInfo(
                    interface_id=row['interface_id'],
                    name=row['name'],
                    url=row['url'],
                    method=row['method'],
                    description=row['description'] or "",
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_check=datetime.fromisoformat(row['last_check']) 
                        if row['last_check'] else None,
                    status=InterfaceStatus(row['status']),
                    health_score=row['health_score']
                )
                
                # 更新缓存
                self._interface_cache[interface_id] = interface
                return interface
            
            return None
        except Exception as e:
            logger.error(f"获取接口信息失败: {e}")
            return None
    
    def get_all_interfaces(self) -> List[InterfaceInfo]:
        """获取所有接口"""
        try:
            query = "SELECT * FROM interfaces"
            result = self.db_manager.execute_query(query)
            
            interfaces = []
            for row in result:
                interface = InterfaceInfo(
                    interface_id=row['interface_id'],
                    name=row['name'],
                    url=row['url'],
                    method=row['method'],
                    description=row['description'] or "",
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_check=datetime.fromisoformat(row['last_check']) 
                        if row['last_check'] else None,
                    status=InterfaceStatus(row['status']),
                    health_score=row['health_score']
                )
                interfaces.append(interface)
            
            return interfaces
        except Exception as e:
            logger.error(f"获取所有接口失败: {e}")
            return []
    
    def save_metrics(self, metrics: InterfaceMetrics) -> bool:
        """保存性能指标"""
        try:
            query = """
                INSERT INTO metrics 
                (interface_id, timestamp, response_time, throughput, error_rate,
                 availability, cpu_usage, memory_usage, network_latency, success_rate,
                 request_count, error_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                metrics.interface_id, metrics.timestamp.isoformat(),
                metrics.response_time, metrics.throughput, metrics.error_rate,
                metrics.availability, metrics.cpu_usage, metrics.memory_usage,
                metrics.network_latency, metrics.success_rate,
                metrics.request_count, metrics.error_count
            )
            return self.db_manager.execute_update(query, params)
        except Exception as e:
            logger.error(f"保存指标失败: {e}")
            return False
    
    def get_metrics(self, interface_id: str, hours: int = 24) -> List[InterfaceMetrics]:
        """获取性能指标"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            query = """
                SELECT * FROM metrics 
                WHERE interface_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """
            result = self.db_manager.execute_query(query, (interface_id, start_time.isoformat()))
            
            metrics_list = []
            for row in result:
                metrics = InterfaceMetrics(
                    interface_id=row['interface_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    response_time=row['response_time'] or 0.0,
                    throughput=row['throughput'] or 0.0,
                    error_rate=row['error_rate'] or 0.0,
                    availability=row['availability'] or 100.0,
                    cpu_usage=row['cpu_usage'] or 0.0,
                    memory_usage=row['memory_usage'] or 0.0,
                    network_latency=row['network_latency'] or 0.0,
                    success_rate=row['success_rate'] or 100.0,
                    request_count=row['request_count'] or 0,
                    error_count=row['error_count'] or 0
                )
                metrics_list.append(metrics)
            
            return metrics_list
        except Exception as e:
            logger.error(f"获取指标失败: {e}")
            return []
    
    def create_alert_rule(self, rule: AlertRule) -> bool:
        """创建告警规则"""
        return self.alert_manager.create_alert_rule(rule)
    
    def get_alert_rules(self, interface_id: str = None) -> List[AlertRule]:
        """获取告警规则"""
        try:
            if interface_id:
                query = "SELECT * FROM alert_rules WHERE interface_id = ?"
                params = (interface_id,)
            else:
                query = "SELECT * FROM alert_rules"
                params = ()
            
            result = self.db_manager.execute_query(query, params)
            
            rules = []
            for row in result:
                rule = AlertRule(
                    rule_id=row['rule_id'],
                    name=row['name'],
                    interface_id=row['interface_id'],
                    metric=row['metric'],
                    condition=row['condition'],
                    threshold=row['threshold'],
                    level=AlertLevel(row['level']),
                    enabled=bool(row['enabled']),
                    cooldown=row['cooldown'],
                    last_triggered=datetime.fromisoformat(row['last_triggered']) 
                        if row['last_triggered'] else None,
                    created_at=datetime.fromisoformat(row['created_at'])
                )
                rules.append(rule)
            
            return rules
        except Exception as e:
            logger.error(f"获取告警规则失败: {e}")
            return []
    
    def get_alerts(self, interface_id: str = None, resolved: bool = None, 
                  limit: int = 100) -> List[Alert]:
        """获取告警"""
        try:
            conditions = []
            params = []
            
            if interface_id:
                conditions.append("a.interface_id = ?")
                params.append(interface_id)
            
            if resolved is not None:
                conditions.append("resolved = ?")
                params.append(resolved)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"""
                SELECT a.*, i.name as interface_name
                FROM alerts a
                JOIN interfaces i ON a.interface_id = i.interface_id
                WHERE {where_clause}
                ORDER BY a.timestamp DESC
                LIMIT ?
            """
            params.append(limit)
            
            result = self.db_manager.execute_query(query, params)
            
            alerts = []
            for row in result:
                alert = Alert(
                    alert_id=row['alert_id'],
                    rule_id=row['rule_id'],
                    interface_id=row['interface_id'],
                    level=AlertLevel(row['level']),
                    message=row['message'],
                    metric_value=row['metric_value'],
                    threshold=row['threshold'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    resolved=bool(row['resolved']),
                    resolved_at=datetime.fromisoformat(row['resolved_at']) 
                        if row['resolved_at'] else None
                )
                alerts.append(alert)
            
            return alerts
        except Exception as e:
            logger.error(f"获取告警失败: {e}")
            return []
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        try:
            query = """
                UPDATE alerts 
                SET resolved = 1, resolved_at = ?
                WHERE alert_id = ?
            """
            params = (datetime.now().isoformat(), alert_id)
            return self.db_manager.execute_update(query, params)
        except Exception as e:
            logger.error(f"解决告警失败: {e}")
            return False
    
    def get_health_assessment(self, interface_id: str) -> Dict[str, Any]:
        """获取健康评估"""
        try:
            # 获取最近的指标
            recent_metrics = self.get_metrics(interface_id, hours=24)
            if not recent_metrics:
                return {'error': 'no_metrics_available'}
            
            # 计算健康分数
            latest_metrics = recent_metrics[0]
            health_score = self.health_calculator.calculate_health_score(latest_metrics)
            health_status = self.health_calculator.get_health_status(health_score)
            
            # 获取性能聚合数据
            performance_data = self.performance_aggregator.aggregate_metrics(
                interface_id, timedelta(hours=24)
            )
            
            # 获取趋势分析
            trend_data = self.performance_aggregator.get_trend_analysis(interface_id, 7)
            
            return {
                'interface_id': interface_id,
                'current_health_score': health_score,
                'current_health_status': health_status.value,
                'latest_metrics': latest_metrics.to_dict(),
                'performance_summary': performance_data,
                'trend_analysis': trend_data,
                'recommendations': self._generate_health_recommendations(health_score, performance_data)
            }
        except Exception as e:
            logger.error(f"获取健康评估失败: {e}")
            return {'error': str(e)}
    
    def _generate_health_recommendations(self, health_score: float, 
                                       performance_data: Dict[str, float]) -> List[str]:
        """生成健康建议"""
        recommendations = []
        
        try:
            if health_score < 60:
                recommendations.append("健康分数较低，建议立即进行系统检查和优化")
            
            if performance_data.get('avg_response_time', 0) > 1000:
                recommendations.append("响应时间过长，建议检查网络延迟和服务器性能")
            
            if performance_data.get('avg_error_rate', 0) > 5:
                recommendations.append("错误率较高，建议检查错误日志和异常处理")
            
            if performance_data.get('avg_availability', 100) < 95:
                recommendations.append("可用性较低，建议检查系统稳定性和故障恢复能力")
            
            if performance_data.get('avg_cpu_usage', 0) > 80:
                recommendations.append("CPU使用率过高，建议优化代码或增加硬件资源")
            
            if performance_data.get('avg_memory_usage', 0) > 80:
                recommendations.append("内存使用率过高，建议检查内存泄漏或增加内存")
            
            if not recommendations:
                recommendations.append("系统健康状况良好，建议继续保持当前状态")
        
        except Exception as e:
            logger.error(f"生成健康建议失败: {e}")
            recommendations.append("生成建议时出现错误，建议手动检查系统状态")
        
        return recommendations
    
    def predict_maintenance(self, interface_id: str, days_ahead: int = 7) -> Dict[str, Any]:
        """预测性维护"""
        return self.predictive_analyzer.predict_failure_risk(interface_id, days_ahead)
    
    def get_dashboard_data(self, interface_ids: List[str] = None) -> Dict[str, Any]:
        """获取仪表板数据"""
        return self.visualization_engine.create_dashboard_data(interface_ids)
    
    def generate_performance_chart(self, interface_id: str, days: int = 7) -> str:
        """生成性能图表"""
        return self.visualization_engine.generate_performance_chart(interface_id, days)
    
    def generate_status_report(self, interface_ids: List[str] = None, 
                             days: int = 7) -> Dict[str, Any]:
        """生成状态报告"""
        return self.report_generator.generate_status_report(interface_ids, days)
    
    def export_data(self, interface_ids: List[str] = None, 
                   format: str = 'json') -> str:
        """导出数据"""
        try:
            if not interface_ids:
                interfaces = self.get_all_interfaces()
                interface_ids = [i.interface_id for i in interfaces]
            
            export_data = {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'interface_count': len(interface_ids),
                    'format': format
                },
                'interfaces': [],
                'metrics': [],
                'alerts': [],
                'rules': []
            }
            
            # 导出接口信息
            for interface_id in interface_ids:
                interface = self.get_interface(interface_id)
                if interface:
                    export_data['interfaces'].append(interface.to_dict())
            
            # 导出指标数据
            for interface_id in interface_ids:
                metrics = self.get_metrics(interface_id, hours=168)  # 一周数据
                for metric in metrics:
                    export_data['metrics'].append(metric.to_dict())
            
            # 导出告警数据
            for interface_id in interface_ids:
                alerts = self.get_alerts(interface_id, limit=1000)
                for alert in alerts:
                    export_data['alerts'].append(alert.to_dict())
            
            # 导出告警规则
            for interface_id in interface_ids:
                rules = self.get_alert_rules(interface_id)
                for rule in rules:
                    rule_dict = asdict(rule)
                    rule_dict['level'] = rule.level.value  # 转换枚举为字符串
                    export_data['rules'].append(rule_dict)
            
            # 保存到文件
            timestamp = int(time.time())
            filename = f"interface_export_{timestamp}.{format}"
            
            if format.lower() == 'json':
                # 自定义JSON编码器，处理datetime对象
                def json_serializer(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, Enum):
                        return obj.value
                    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2, default=json_serializer)
            elif format.lower() == 'csv':
                # 简化的CSV导出，只包含主要指标
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'interface_id', 'timestamp', 'response_time', 'throughput',
                        'error_rate', 'availability', 'health_score'
                    ])
                    for metric_data in export_data['metrics']:
                        writer.writerow([
                            metric_data['interface_id'],
                            metric_data['timestamp'],
                            metric_data['response_time'],
                            metric_data['throughput'],
                            metric_data['error_rate'],
                            metric_data['availability'],
                            ''  # 健康分数在metrics表中没有，需要单独查询
                        ])
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logger.info(f"数据导出完成: {filename}")
            return filename
        
        except Exception as e:
            logger.error(f"导出数据失败: {e}")
            raise
    
    def add_status_change_callback(self, callback: Callable[[str, InterfaceStatus, InterfaceStatus], None]):
        """添加状态变更回调"""
        self.status_change_callbacks.append(callback)
        self.state_monitor.add_state_callback(callback)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
        self.alert_manager.add_alert_callback(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统整体状态"""
        try:
            interfaces = self.get_all_interfaces()
            total_interfaces = len(interfaces)
            
            if total_interfaces == 0:
                return {
                    'status': 'no_interfaces',
                    'message': '没有注册任何接口',
                    'monitoring_active': self.monitoring_active
                }
            
            # 统计状态分布
            status_counts = defaultdict(int)
            health_scores = []
            
            for interface in interfaces:
                status_counts[interface.status.value] += 1
                health_scores.append(interface.health_score)
            
            # 计算整体健康分数
            avg_health_score = statistics.mean(health_scores) if health_scores else 0
            
            # 获取活跃告警数量
            active_alerts = self.get_alerts(resolved=False, limit=1000)
            critical_alerts = [a for a in active_alerts if a.level == AlertLevel.CRITICAL]
            
            # 确定系统状态
            if critical_alerts:
                system_status = 'critical'
            elif status_counts.get('offline', 0) > total_interfaces * 0.1:
                system_status = 'warning'
            elif avg_health_score < 70:
                system_status = 'degraded'
            elif avg_health_score >= 85:
                system_status = 'excellent'
            else:
                system_status = 'good'
            
            return {
                'status': system_status,
                'monitoring_active': self.monitoring_active,
                'total_interfaces': total_interfaces,
                'status_distribution': dict(status_counts),
                'average_health_score': round(avg_health_score, 2),
                'active_alerts': len(active_alerts),
                'critical_alerts': len(critical_alerts),
                'last_check': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'monitoring_active': self.monitoring_active
            }
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """清理旧数据"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.isoformat()
            
            # 清理旧指标数据
            metrics_query = "DELETE FROM metrics WHERE timestamp < ?"
            metrics_result = self.db_manager.execute_update(metrics_query, (cutoff_str,))
            
            # 清理已解决的旧告警
            alerts_query = "DELETE FROM alerts WHERE resolved = 1 AND resolved_at < ?"
            alerts_result = self.db_manager.execute_update(alerts_query, (cutoff_str,))
            
            # 清理历史分析数据
            analysis_query = "DELETE FROM historical_analysis WHERE analysis_date < ?"
            analysis_result = self.db_manager.execute_update(analysis_query, (cutoff_str,))
            
            logger.info(f"清理完成: 指标数据={metrics_result}, 告警数据={alerts_result}, 分析数据={analysis_result}")
            
            return {
                'metrics_deleted': metrics_result,
                'alerts_deleted': alerts_result,
                'analysis_deleted': analysis_result
            }
        
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
            return {'error': str(e)}
    
    def close(self):
        """关闭聚合器"""
        try:
            # 停止监控
            if self.monitoring_active:
                asyncio.create_task(self.stop_monitoring())
            
            # 关闭线程池
            self.executor.shutdown(wait=True)
            
            logger.info("接口状态聚合器已关闭")
        
        except Exception as e:
            logger.error(f"关闭聚合器失败: {e}")


# 辅助函数和工具
def create_sample_interface_checker() -> Callable[[InterfaceInfo], InterfaceMetrics]:
    """创建示例接口检查器"""
    async def checker(interface: InterfaceInfo) -> InterfaceMetrics:
        """模拟接口检查"""
        import random
        
        # 模拟网络请求延迟
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # 生成模拟指标
        response_time = random.uniform(50, 500)
        throughput = random.uniform(100, 1000)
        error_rate = random.uniform(0, 5)
        availability = random.uniform(95, 100)
        cpu_usage = random.uniform(10, 80)
        memory_usage = random.uniform(20, 70)
        network_latency = random.uniform(10, 100)
        success_rate = 100 - error_rate
        request_count = random.randint(100, 1000)
        error_count = int(request_count * error_rate / 100)
        
        return InterfaceMetrics(
            interface_id=interface.interface_id,
            timestamp=datetime.now(),
            response_time=response_time,
            throughput=throughput,
            error_rate=error_rate,
            availability=availability,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            network_latency=network_latency,
            success_rate=success_rate,
            request_count=request_count,
            error_count=error_count
        )
    
    return checker


# 示例使用代码
async def example_usage():
    """示例使用"""
    # 创建聚合器
    aggregator = InterfaceStateAggregator()
    
    # 注册示例接口
    interface = InterfaceInfo(
        interface_id="api_example_001",
        name="示例API接口",
        url="https://api.example.com/health",
        method="GET",
        description="用于演示的示例接口",
        tags=["demo", "example"]
    )
    
    checker = create_sample_interface_checker()
    aggregator.register_interface(interface, checker)
    
    # 创建告警规则
    alert_rule = AlertRule(
        rule_id="response_time_high",
        name="响应时间过高",
        interface_id="api_example_001",
        metric="response_time",
        condition=">",
        threshold=300,
        level=AlertLevel.WARNING
    )
    aggregator.create_alert_rule(alert_rule)
    
    # 开始监控
    await aggregator.start_monitoring(interval=30)
    
    # 等待一段时间收集数据
    await asyncio.sleep(60)
    
    # 获取仪表板数据
    dashboard_data = aggregator.get_dashboard_data()
    print("仪表板数据:", json.dumps(dashboard_data, ensure_ascii=False, indent=2))
    
    # 生成状态报告
    report = aggregator.generate_status_report(days=1)
    print("状态报告:", json.dumps(report, ensure_ascii=False, indent=2))
    
    # 预测性维护
    prediction = aggregator.predict_maintenance("api_example_001")
    print("预测性维护:", json.dumps(prediction, ensure_ascii=False, indent=2))
    
    # 停止监控
    await aggregator.stop_monitoring()
    
    # 关闭聚合器
    aggregator.close()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())