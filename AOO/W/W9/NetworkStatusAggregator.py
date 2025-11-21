#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W9网络状态聚合器

网络状态聚合器是一个综合性的网络监控系统，提供：
- 状态收集器：从各个网络模块收集状态信息
- 数据聚合：聚合多个网络模块的结果
- 状态分析：分析网络状态和趋势
- 报告生成：生成综合网络状态报告
- 状态监控：实时监控网络状态
- 预警机制：网络异常时预警
- 历史记录：保存历史网络状态
- 仪表板：提供可视化的网络状态仪表板

作者: W9网络状态聚合器团队
版本: 1.0.0
创建时间: 2025-11-06
"""

import json
import time
import threading
import logging
import sqlite3
import subprocess
import psutil
import socket
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


@dataclass
class NetworkModuleStatus:
    """网络模块状态数据结构"""
    module_name: str
    status: str  # 'healthy', 'warning', 'critical', 'offline'
    response_time: float
    uptime: float
    cpu_usage: float
    memory_usage: float
    error_rate: float
    timestamp: datetime
    details: Dict[str, Any]


@dataclass
class NetworkAlert:
    """网络预警数据结构"""
    alert_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    module_name: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class StatusCollector:
    """状态收集器 - 从各个网络模块收集状态信息"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.modules = {}
        self.collectors = {
            'ping': self._ping_collector,
            'http': self._http_collector,
            'tcp': self._tcp_collector,
            'system': self._system_collector,
            'custom': self._custom_collector
        }
    
    def register_module(self, module_name: str, collector_type: str, config: Dict[str, Any]):
        """注册网络模块"""
        self.modules[module_name] = {
            'type': collector_type,
            'config': config,
            'last_status': None,
            'last_check': None
        }
        self.logger.info(f"已注册模块: {module_name} ({collector_type})")
    
    def collect_status(self, module_name: str) -> NetworkModuleStatus:
        """收集指定模块的状态"""
        if module_name not in self.modules:
            raise ValueError(f"模块 {module_name} 未注册")
        
        module_info = self.modules[module_name]
        collector_type = module_info['type']
        config = module_info['config']
        
        # 执行相应的收集器
        if collector_type in self.collectors:
            status = self.collectors[collector_type](module_name, config)
            module_info['last_status'] = status
            module_info['last_check'] = datetime.now()
            return status
        else:
            raise ValueError(f"未知的收集器类型: {collector_type}")
    
    def collect_all_status(self) -> Dict[str, NetworkModuleStatus]:
        """收集所有模块的状态"""
        results = {}
        for module_name in self.modules:
            try:
                results[module_name] = self.collect_status(module_name)
            except Exception as e:
                self.logger.error(f"收集模块 {module_name} 状态失败: {e}")
                # 创建错误状态
                results[module_name] = NetworkModuleStatus(
                    module_name=module_name,
                    status='offline',
                    response_time=0.0,
                    uptime=0.0,
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    error_rate=100.0,
                    timestamp=datetime.now(),
                    details={'error': str(e)}
                )
        return results
    
    def _ping_collector(self, module_name: str, config: Dict[str, Any]) -> NetworkModuleStatus:
        """Ping收集器"""
        target = config.get('target', '8.8.8.8')
        try:
            # 执行ping命令
            result = subprocess.run(
                ['ping', '-c', '1', '-W', '3', target],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # 解析ping结果
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'time=' in line:
                        time_str = line.split('time=')[1].split()[0]
                        response_time = float(time_str)
                        break
                else:
                    response_time = 0.0
                
                status = 'healthy' if response_time < 100 else 'warning'
            else:
                response_time = 0.0
                status = 'critical'
            
            return NetworkModuleStatus(
                module_name=module_name,
                status=status,
                response_time=response_time,
                uptime=100.0,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                error_rate=0.0 if status == 'healthy' else 50.0,
                timestamp=datetime.now(),
                details={'target': target, 'ping_result': result.stdout}
            )
        except Exception as e:
            return NetworkModuleStatus(
                module_name=module_name,
                status='offline',
                response_time=0.0,
                uptime=0.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                error_rate=100.0,
                timestamp=datetime.now(),
                details={'error': str(e)}
            )
    
    def _http_collector(self, module_name: str, config: Dict[str, Any]) -> NetworkModuleStatus:
        """HTTP收集器"""
        url = config.get('url', 'http://httpbin.org/get')
        timeout = config.get('timeout', 5)
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout)
            response_time = (time.time() - start_time) * 1000
            
            status_code = response.status_code
            if 200 <= status_code < 300:
                status = 'healthy' if response_time < 1000 else 'warning'
                error_rate = 0.0
            elif 400 <= status_code < 500:
                status = 'warning'
                error_rate = 25.0
            else:
                status = 'critical'
                error_rate = 50.0
            
            return NetworkModuleStatus(
                module_name=module_name,
                status=status,
                response_time=response_time,
                uptime=100.0 if status == 'healthy' else 75.0,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                error_rate=error_rate,
                timestamp=datetime.now(),
                details={
                    'url': url,
                    'status_code': status_code,
                    'response_headers': dict(response.headers)
                }
            )
        except Exception as e:
            return NetworkModuleStatus(
                module_name=module_name,
                status='offline',
                response_time=0.0,
                uptime=0.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                error_rate=100.0,
                timestamp=datetime.now(),
                details={'error': str(e)}
            )
    
    def _tcp_collector(self, module_name: str, config: Dict[str, Any]) -> NetworkModuleStatus:
        """TCP连接收集器"""
        host = config.get('host', 'localhost')
        port = config.get('port', 80)
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            start_time = time.time()
            result = sock.connect_ex((host, port))
            response_time = (time.time() - start_time) * 1000
            sock.close()
            
            if result == 0:
                status = 'healthy' if response_time < 500 else 'warning'
                error_rate = 0.0
            else:
                status = 'critical'
                error_rate = 100.0
            
            return NetworkModuleStatus(
                module_name=module_name,
                status=status,
                response_time=response_time,
                uptime=100.0 if status == 'healthy' else 0.0,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                error_rate=error_rate,
                timestamp=datetime.now(),
                details={'host': host, 'port': port, 'connection_result': result}
            )
        except Exception as e:
            return NetworkModuleStatus(
                module_name=module_name,
                status='offline',
                response_time=0.0,
                uptime=0.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                error_rate=100.0,
                timestamp=datetime.now(),
                details={'error': str(e)}
            )
    
    def _system_collector(self, module_name: str, config: Dict[str, Any]) -> NetworkModuleStatus:
        """系统资源收集器"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 计算系统健康状态
            if cpu_usage < 70 and memory.percent < 80:
                status = 'healthy'
                error_rate = 0.0
            elif cpu_usage < 85 and memory.percent < 90:
                status = 'warning'
                error_rate = 10.0
            else:
                status = 'critical'
                error_rate = 25.0
            
            return NetworkModuleStatus(
                module_name=module_name,
                status=status,
                response_time=1.0,  # 系统检查响应时间
                uptime=psutil.boot_time(),  # 系统运行时间
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                error_rate=error_rate,
                timestamp=datetime.now(),
                details={
                    'cpu_count': psutil.cpu_count(),
                    'memory_total': memory.total,
                    'memory_available': memory.available,
                    'disk_total': disk.total,
                    'disk_free': disk.free
                }
            )
        except Exception as e:
            return NetworkModuleStatus(
                module_name=module_name,
                status='offline',
                response_time=0.0,
                uptime=0.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                error_rate=100.0,
                timestamp=datetime.now(),
                details={'error': str(e)}
            )
    
    def _custom_collector(self, module_name: str, config: Dict[str, Any]) -> NetworkModuleStatus:
        """自定义收集器"""
        collector_func = config.get('collector_func')
        if not collector_func or not callable(collector_func):
            raise ValueError("自定义收集器需要提供collector_func")
        
        try:
            result = collector_func()
            return NetworkModuleStatus(
                module_name=module_name,
                status=result.get('status', 'healthy'),
                response_time=result.get('response_time', 0.0),
                uptime=result.get('uptime', 100.0),
                cpu_usage=result.get('cpu_usage', 0.0),
                memory_usage=result.get('memory_usage', 0.0),
                error_rate=result.get('error_rate', 0.0),
                timestamp=datetime.now(),
                details=result.get('details', {})
            )
        except Exception as e:
            return NetworkModuleStatus(
                module_name=module_name,
                status='offline',
                response_time=0.0,
                uptime=0.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                error_rate=100.0,
                timestamp=datetime.now(),
                details={'error': str(e)}
            )


class DataAggregator:
    """数据聚合器 - 聚合多个网络模块的结果"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def aggregate_status(self, status_dict: Dict[str, NetworkModuleStatus]) -> Dict[str, Any]:
        """聚合多个模块的状态"""
        if not status_dict:
            return self._empty_aggregation()
        
        # 统计各种状态的数量
        status_counts = defaultdict(int)
        total_response_time = 0.0
        total_uptime = 0.0
        total_cpu = 0.0
        total_memory = 0.0
        total_error_rate = 0.0
        module_count = len(status_dict)
        
        for status in status_dict.values():
            status_counts[status.status] += 1
            total_response_time += status.response_time
            total_uptime += status.uptime
            total_cpu += status.cpu_usage
            total_memory += status.memory_usage
            total_error_rate += status.error_rate
        
        # 计算平均值
        avg_response_time = total_response_time / module_count if module_count > 0 else 0
        avg_uptime = total_uptime / module_count if module_count > 0 else 0
        avg_cpu = total_cpu / module_count if module_count > 0 else 0
        avg_memory = total_memory / module_count if module_count > 0 else 0
        avg_error_rate = total_error_rate / module_count if module_count > 0 else 0
        
        # 确定整体健康状态
        overall_status = self._determine_overall_status(status_counts, avg_error_rate)
        
        # 计算健康度评分
        health_score = self._calculate_health_score(status_counts, avg_response_time, avg_error_rate)
        
        return {
            'overall_status': overall_status,
            'health_score': health_score,
            'module_count': module_count,
            'status_distribution': dict(status_counts),
            'metrics': {
                'avg_response_time': avg_response_time,
                'avg_uptime': avg_uptime,
                'avg_cpu_usage': avg_cpu,
                'avg_memory_usage': avg_memory,
                'avg_error_rate': avg_error_rate
            },
            'timestamp': datetime.now(),
            'details': {
                'healthy_modules': status_counts.get('healthy', 0),
                'warning_modules': status_counts.get('warning', 0),
                'critical_modules': status_counts.get('critical', 0),
                'offline_modules': status_counts.get('offline', 0)
            }
        }
    
    def _empty_aggregation(self) -> Dict[str, Any]:
        """空聚合结果"""
        return {
            'overall_status': 'unknown',
            'health_score': 0.0,
            'module_count': 0,
            'status_distribution': {},
            'metrics': {
                'avg_response_time': 0.0,
                'avg_uptime': 0.0,
                'avg_cpu_usage': 0.0,
                'avg_memory_usage': 0.0,
                'avg_error_rate': 0.0
            },
            'timestamp': datetime.now(),
            'details': {
                'healthy_modules': 0,
                'warning_modules': 0,
                'critical_modules': 0,
                'offline_modules': 0
            }
        }
    
    def _determine_overall_status(self, status_counts: Dict[str, int], avg_error_rate: float) -> str:
        """确定整体健康状态"""
        if status_counts.get('offline', 0) > 0:
            return 'critical'
        elif status_counts.get('critical', 0) > 0:
            return 'critical'
        elif status_counts.get('warning', 0) > 0:
            return 'warning'
        elif avg_error_rate > 5.0:
            return 'warning'
        else:
            return 'healthy'
    
    def _calculate_health_score(self, status_counts: Dict[str, int], avg_response_time: float, avg_error_rate: float) -> float:
        """计算健康度评分 (0-100)"""
        score = 100.0
        
        # 根据状态扣分
        score -= status_counts.get('warning', 0) * 10
        score -= status_counts.get('critical', 0) * 30
        score -= status_counts.get('offline', 0) * 50
        
        # 根据响应时间扣分
        if avg_response_time > 1000:
            score -= 20
        elif avg_response_time > 500:
            score -= 10
        elif avg_response_time > 200:
            score -= 5
        
        # 根据错误率扣分
        score -= avg_error_rate
        
        return max(0.0, min(100.0, score))


class StatusAnalyzer:
    """状态分析器 - 分析网络状态和趋势"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.history_data = deque(maxlen=1000)  # 保留最近1000条记录
    
    def add_status_record(self, aggregation_result: Dict[str, Any]):
        """添加状态记录到历史数据"""
        self.history_data.append({
            'timestamp': aggregation_result['timestamp'],
            'overall_status': aggregation_result['overall_status'],
            'health_score': aggregation_result['health_score'],
            'metrics': aggregation_result['metrics']
        })
    
    def analyze_trends(self, hours: int = 24) -> Dict[str, Any]:
        """分析网络状态趋势"""
        if len(self.history_data) < 2:
            return {'error': '历史数据不足，无法分析趋势'}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [record for record in self.history_data 
                      if record['timestamp'] >= cutoff_time]
        
        if len(recent_data) < 2:
            return {'error': f'最近{hours}小时数据不足，无法分析趋势'}
        
        # 分析健康度趋势
        health_scores = [record['health_score'] for record in recent_data]
        health_trend = self._analyze_trend(health_scores)
        
        # 分析响应时间趋势
        response_times = [record['metrics']['avg_response_time'] for record in recent_data]
        response_trend = self._analyze_trend(response_times)
        
        # 分析错误率趋势
        error_rates = [record['metrics']['avg_error_rate'] for record in recent_data]
        error_trend = self._analyze_trend(error_rates)
        
        # 计算统计信息
        stats = {
            'health_score': {
                'min': min(health_scores),
                'max': max(health_scores),
                'avg': sum(health_scores) / len(health_scores),
                'current': health_scores[-1] if health_scores else 0
            },
            'response_time': {
                'min': min(response_times),
                'max': max(response_times),
                'avg': sum(response_times) / len(response_times),
                'current': response_times[-1] if response_times else 0
            },
            'error_rate': {
                'min': min(error_rates),
                'max': max(error_rates),
                'avg': sum(error_rates) / len(error_rates),
                'current': error_rates[-1] if error_rates else 0
            }
        }
        
        return {
            'analysis_period': f'{hours}小时',
            'data_points': len(recent_data),
            'trends': {
                'health_score_trend': health_trend,
                'response_time_trend': response_trend,
                'error_rate_trend': error_trend
            },
            'statistics': stats,
            'recommendations': self._generate_recommendations(stats, health_trend, response_trend)
        }
    
    def _analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """分析数值趋势"""
        if len(values) < 2:
            return {'direction': 'unknown', 'slope': 0.0, 'confidence': 0.0}
        
        # 计算线性回归斜率
        x = list(range(len(values)))
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # 确定趋势方向
        if abs(slope) < 0.1:
            direction = 'stable'
        elif slope > 0:
            direction = 'improving' if values[-1] > values[0] else 'degrading'
        else:
            direction = 'degrading' if values[-1] < values[0] else 'improving'
        
        # 计算置信度（基于变化幅度）
        variation = max(values) - min(values)
        confidence = min(100.0, variation / (sum(values) / len(values)) * 100) if sum(values) > 0 else 0
        
        return {
            'direction': direction,
            'slope': slope,
            'confidence': confidence,
            'change_percent': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
        }
    
    def _generate_recommendations(self, stats: Dict, health_trend: Dict, response_trend: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于健康度评分
        if stats['health_score']['current'] < 60:
            recommendations.append("网络健康度较低，建议检查所有模块状态")
        
        # 基于响应时间
        if stats['response_time']['current'] > 1000:
            recommendations.append("响应时间过长，建议优化网络配置")
        
        # 基于错误率
        if stats['error_rate']['current'] > 5:
            recommendations.append("错误率较高，建议检查网络连接稳定性")
        
        # 基于趋势
        if health_trend['direction'] == 'degrading':
            recommendations.append("网络健康度呈下降趋势，建议加强监控")
        
        if response_trend['direction'] == 'degrading':
            recommendations.append("响应时间呈上升趋势，建议检查网络负载")
        
        if not recommendations:
            recommendations.append("网络状态良好，继续保持当前配置")
        
        return recommendations


class AlertManager:
    """预警管理器 - 网络异常时预警"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alerts = []
        self.alert_rules = []
        self.alert_callbacks = []
    
    def add_alert_rule(self, rule_name: str, condition: Callable[[Dict[str, Any]], bool], 
                      severity: str, message_template: str):
        """添加预警规则"""
        self.alert_rules.append({
            'name': rule_name,
            'condition': condition,
            'severity': severity,
            'message_template': message_template
        })
        self.logger.info(f"已添加预警规则: {rule_name}")
    
    def check_alerts(self, aggregation_result: Dict[str, Any]) -> List[NetworkAlert]:
        """检查预警条件"""
        new_alerts = []
        
        for rule in self.alert_rules:
            try:
                if rule['condition'](aggregation_result):
                    alert = NetworkAlert(
                        alert_id=f"{rule['name']}_{int(time.time())}",
                        severity=rule['severity'],
                        message=rule['message_template'].format(**aggregation_result),
                        module_name='system',
                        timestamp=datetime.now()
                    )
                    new_alerts.append(alert)
                    self.alerts.append(alert)
                    self.logger.warning(f"触发预警: {alert.message}")
            except Exception as e:
                self.logger.error(f"检查预警规则 {rule['name']} 时出错: {e}")
        
        # 执行回调函数
        for callback in self.alert_callbacks:
            try:
                callback(new_alerts)
            except Exception as e:
                self.logger.error(f"执行预警回调时出错: {e}")
        
        return new_alerts
    
    def add_alert_callback(self, callback: Callable[[List[NetworkAlert]], None]):
        """添加预警回调函数"""
        self.alert_callbacks.append(callback)
    
    def resolve_alert(self, alert_id: str):
        """解决预警"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                self.logger.info(f"已解决预警: {alert_id}")
                break
    
    def get_active_alerts(self) -> List[NetworkAlert]:
        """获取活跃预警"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alert_history(self, hours: int = 24) -> List[NetworkAlert]:
        """获取预警历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]


class HistoryManager:
    """历史记录管理器 - 保存历史网络状态"""
    
    def __init__(self, db_path: str = "network_status.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建状态历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS status_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    overall_status TEXT,
                    health_score REAL,
                    module_count INTEGER,
                    avg_response_time REAL,
                    avg_uptime REAL,
                    avg_cpu_usage REAL,
                    avg_memory_usage REAL,
                    avg_error_rate REAL,
                    details TEXT
                )
            ''')
            
            # 创建预警历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT,
                    severity TEXT,
                    message TEXT,
                    module_name TEXT,
                    timestamp DATETIME,
                    resolved BOOLEAN,
                    resolution_time DATETIME
                )
            ''')
            
            # 创建模块状态历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS module_status_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    module_name TEXT,
                    status TEXT,
                    response_time REAL,
                    uptime REAL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    error_rate REAL,
                    timestamp DATETIME,
                    details TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("数据库初始化完成")
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
    
    def save_status_record(self, aggregation_result: Dict[str, Any], module_statuses: Dict[str, NetworkModuleStatus]):
        """保存状态记录"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 保存聚合结果
            cursor.execute('''
                INSERT INTO status_history 
                (timestamp, overall_status, health_score, module_count, 
                 avg_response_time, avg_uptime, avg_cpu_usage, avg_memory_usage, 
                 avg_error_rate, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                aggregation_result['timestamp'],
                aggregation_result['overall_status'],
                aggregation_result['health_score'],
                aggregation_result['module_count'],
                aggregation_result['metrics']['avg_response_time'],
                aggregation_result['metrics']['avg_uptime'],
                aggregation_result['metrics']['avg_cpu_usage'],
                aggregation_result['metrics']['avg_memory_usage'],
                aggregation_result['metrics']['avg_error_rate'],
                json.dumps(aggregation_result['details'])
            ))
            
            # 保存模块状态
            for module_name, status in module_statuses.items():
                cursor.execute('''
                    INSERT INTO module_status_history 
                    (module_name, status, response_time, uptime, cpu_usage, 
                     memory_usage, error_rate, timestamp, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    module_name,
                    status.status,
                    status.response_time,
                    status.uptime,
                    status.cpu_usage,
                    status.memory_usage,
                    status.error_rate,
                    status.timestamp,
                    json.dumps(status.details)
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"保存状态记录失败: {e}")
    
    def save_alert(self, alert: NetworkAlert):
        """保存预警记录"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alert_history 
                (alert_id, severity, message, module_name, timestamp, resolved, resolution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.severity,
                alert.message,
                alert.module_name,
                alert.timestamp,
                alert.resolved,
                alert.resolution_time
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"保存预警记录失败: {e}")
    
    def get_status_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取状态历史"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cursor.execute('''
                SELECT * FROM status_history 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
            
            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                result['details'] = json.loads(result['details']) if result['details'] else {}
                results.append(result)
            
            conn.close()
            return results
        except Exception as e:
            self.logger.error(f"获取状态历史失败: {e}")
            return []
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取预警历史"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cursor.execute('''
                SELECT * FROM alert_history 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
            
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            conn.close()
            return results
        except Exception as e:
            self.logger.error(f"获取预警历史失败: {e}")
            return []


class ReportGenerator:
    """报告生成器 - 生成综合网络状态报告"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_status_report(self, aggregation_result: Dict[str, Any], 
                             trend_analysis: Dict[str, Any], 
                             active_alerts: List[NetworkAlert]) -> str:
        """生成状态报告"""
        report = []
        report.append("=" * 60)
        report.append("W9网络状态聚合器 - 综合状态报告")
        report.append("=" * 60)
        report.append(f"报告生成时间: {aggregation_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 整体状态
        report.append("【整体网络状态】")
        report.append(f"状态: {aggregation_result['overall_status'].upper()}")
        report.append(f"健康度评分: {aggregation_result['health_score']:.1f}/100")
        report.append(f"监控模块数量: {aggregation_result['module_count']}")
        report.append("")
        
        # 状态分布
        report.append("【模块状态分布】")
        status_dist = aggregation_result['status_distribution']
        for status, count in status_dist.items():
            percentage = (count / aggregation_result['module_count']) * 100
            report.append(f"{status.upper()}: {count} 个模块 ({percentage:.1f}%)")
        report.append("")
        
        # 性能指标
        report.append("【性能指标】")
        metrics = aggregation_result['metrics']
        report.append(f"平均响应时间: {metrics['avg_response_time']:.2f}ms")
        report.append(f"平均运行时间: {metrics['avg_uptime']:.1f}%")
        report.append(f"平均CPU使用率: {metrics['avg_cpu_usage']:.1f}%")
        report.append(f"平均内存使用率: {metrics['avg_memory_usage']:.1f}%")
        report.append(f"平均错误率: {metrics['avg_error_rate']:.2f}%")
        report.append("")
        
        # 趋势分析
        if 'error' not in trend_analysis:
            report.append("【趋势分析】")
            report.append(f"分析周期: {trend_analysis['analysis_period']}")
            report.append(f"数据点数量: {trend_analysis['data_points']}")
            
            trends = trend_analysis['trends']
            report.append(f"健康度趋势: {trends['health_score_trend']['direction']} "
                        f"(变化: {trends['health_score_trend']['change_percent']:.1f}%)")
            report.append(f"响应时间趋势: {trends['response_time_trend']['direction']} "
                        f"(变化: {trends['response_time_trend']['change_percent']:.1f}%)")
            report.append(f"错误率趋势: {trends['error_rate_trend']['direction']} "
                        f"(变化: {trends['error_rate_trend']['change_percent']:.1f}%)")
            report.append("")
            
            # 建议
            report.append("【优化建议】")
            for i, recommendation in enumerate(trend_analysis['recommendations'], 1):
                report.append(f"{i}. {recommendation}")
            report.append("")
        
        # 活跃预警
        if active_alerts:
            report.append("【活跃预警】")
            for alert in active_alerts:
                report.append(f"[{alert.severity.upper()}] {alert.message}")
                report.append(f"  模块: {alert.module_name}, 时间: {alert.timestamp.strftime('%H:%M:%S')}")
            report.append("")
        else:
            report.append("【活跃预警】无活跃预警")
            report.append("")
        
        report.append("=" * 60)
        report.append("报告生成完成")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def export_report_to_file(self, report_content: str, filename: str = None) -> str:
        """导出报告到文件"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"network_status_report_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"报告已导出到文件: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"导出报告失败: {e}")
            raise


class Dashboard:
    """仪表板 - 提供可视化的网络状态仪表板"""
    
    def __init__(self, aggregator: 'NetworkStatusAggregator'):
        self.aggregator = aggregator
        self.logger = logging.getLogger(__name__)
        self.fig = None
        self.axes = None
        self.animation = None
    
    def create_dashboard(self, update_interval: int = 5000):
        """创建仪表板"""
        try:
            plt.style.use('seaborn-v0_8')
            self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
            self.fig.suptitle('W9网络状态聚合器 - 实时监控仪表板', fontsize=16, fontweight='bold')
            
            # 设置子图
            self._setup_subplots()
            
            # 启动动画更新
            self.animation = FuncAnimation(
                self.fig, self._update_dashboard, 
                interval=update_interval, blit=False
            )
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"创建仪表板失败: {e}")
    
    def _setup_subplots(self):
        """设置子图"""
        # 健康度评分图
        ax1 = self.axes[0, 0]
        ax1.set_title('网络健康度评分', fontweight='bold')
        ax1.set_ylabel('评分')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # 响应时间图
        ax2 = self.axes[0, 1]
        ax2.set_title('平均响应时间', fontweight='bold')
        ax2.set_ylabel('响应时间 (ms)')
        ax2.grid(True, alpha=0.3)
        
        # 状态分布图
        ax3 = self.axes[1, 0]
        ax3.set_title('模块状态分布', fontweight='bold')
        
        # 预警状态图
        ax4 = self.axes[1, 1]
        ax4.set_title('活跃预警', fontweight='bold')
        ax4.set_ylabel('预警数量')
        ax4.grid(True, alpha=0.3)
    
    def _update_dashboard(self, frame):
        """更新仪表板数据"""
        try:
            # 获取历史数据
            history_data = self.aggregator.history_manager.get_status_history(hours=1)
            
            if len(history_data) < 2:
                return
            
            # 准备数据
            timestamps = [datetime.fromisoformat(record['timestamp']) for record in history_data]
            health_scores = [record['health_score'] for record in history_data]
            response_times = [record['avg_response_time'] for record in history_data]
            
            # 更新健康度图
            ax1 = self.axes[0, 0]
            ax1.clear()
            ax1.plot(timestamps, health_scores, 'b-', linewidth=2, label='健康度评分')
            ax1.set_title('网络健康度评分', fontweight='bold')
            ax1.set_ylabel('评分')
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 格式化时间轴
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 更新响应时间图
            ax2 = self.axes[0, 1]
            ax2.clear()
            ax2.plot(timestamps, response_times, 'g-', linewidth=2, label='响应时间')
            ax2.set_title('平均响应时间', fontweight='bold')
            ax2.set_ylabel('响应时间 (ms)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 格式化时间轴
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # 更新状态分布图
            ax3 = self.axes[1, 0]
            ax3.clear()
            
            # 获取最新的状态分布
            latest_record = history_data[-1]
            status_dist = {
                'healthy': latest_record['details'].get('healthy_modules', 0),
                'warning': latest_record['details'].get('warning_modules', 0),
                'critical': latest_record['details'].get('critical_modules', 0),
                'offline': latest_record['details'].get('offline_modules', 0)
            }
            
            colors = ['green', 'yellow', 'orange', 'red']
            statuses = list(status_dist.keys())
            counts = list(status_dist.values())
            
            if sum(counts) > 0:
                ax3.pie(counts, labels=statuses, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('模块状态分布', fontweight='bold')
            
            # 更新预警图
            ax4 = self.axes[1, 1]
            ax4.clear()
            
            # 获取活跃预警
            active_alerts = self.aggregator.alert_manager.get_active_alerts()
            alert_counts = defaultdict(int)
            for alert in active_alerts:
                alert_counts[alert.severity] += 1
            
            if alert_counts:
                severities = list(alert_counts.keys())
                counts = list(alert_counts.values())
                colors = ['green', 'yellow', 'orange', 'red']
                color_map = {'low': 'green', 'medium': 'yellow', 'high': 'orange', 'critical': 'red'}
                bar_colors = [color_map.get(sev, 'gray') for sev in severities]
                
                ax4.bar(severities, counts, color=bar_colors, alpha=0.7)
            ax4.set_title('活跃预警', fontweight='bold')
            ax4.set_ylabel('预警数量')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
        except Exception as e:
            self.logger.error(f"更新仪表板失败: {e}")
    
    def export_dashboard(self, filename: str = None) -> str:
        """导出仪表板截图"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"dashboard_{timestamp}.png"
        
        try:
            if self.fig:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                self.logger.info(f"仪表板已导出到文件: {filename}")
                return filename
            else:
                raise ValueError("仪表板尚未创建")
        except Exception as e:
            self.logger.error(f"导出仪表板失败: {e}")
            raise


class NetworkStatusAggregator:
    """网络状态聚合器主类"""
    
    def __init__(self, db_path: str = "network_status.db"):
        # 初始化各个组件
        self.status_collector = StatusCollector()
        self.data_aggregator = DataAggregator()
        self.status_analyzer = StatusAnalyzer()
        self.alert_manager = AlertManager()
        self.history_manager = HistoryManager(db_path)
        self.report_generator = ReportGenerator()
        self.dashboard = Dashboard(self)
        
        # 监控状态
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 30  # 默认30秒监控间隔
        
        # 设置默认预警规则
        self._setup_default_alert_rules()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _setup_default_alert_rules(self):
        """设置默认预警规则"""
        # 健康度低预警
        self.alert_manager.add_alert_rule(
            "low_health_score",
            lambda result: result['health_score'] < 60,
            "high",
            "网络健康度评分过低: {health_score:.1f}"
        )
        
        # 响应时间过长预警
        self.alert_manager.add_alert_rule(
            "high_response_time",
            lambda result: result['metrics']['avg_response_time'] > 1000,
            "medium",
            "平均响应时间过长: {metrics[avg_response_time]:.2f}ms"
        )
        
        # 错误率过高预警
        self.alert_manager.add_alert_rule(
            "high_error_rate",
            lambda result: result['metrics']['avg_error_rate'] > 5.0,
            "high",
            "错误率过高: {metrics[avg_error_rate]:.2f}%"
        )
        
        # 模块离线预警
        self.alert_manager.add_alert_rule(
            "module_offline",
            lambda result: result['details']['offline_modules'] > 0,
            "critical",
            "有 {details[offline_modules]} 个模块离线"
        )
        
        # CPU使用率过高预警
        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            lambda result: result['metrics']['avg_cpu_usage'] > 80,
            "medium",
            "平均CPU使用率过高: {metrics[avg_cpu_usage]:.1f}%"
        )
        
        # 内存使用率过高预警
        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            lambda result: result['metrics']['avg_memory_usage'] > 85,
            "medium",
            "平均内存使用率过高: {metrics[avg_memory_usage]:.1f}%"
        )
    
    def register_module(self, module_name: str, collector_type: str, config: Dict[str, Any]):
        """注册网络模块"""
        self.status_collector.register_module(module_name, collector_type, config)
    
    def start_monitoring(self, interval: int = 30):
        """开始监控"""
        if self.monitoring:
            self.logger.warning("监控已在运行中")
            return
        
        self.monitoring = True
        self.monitor_interval = interval
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"开始监控，间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("已停止监控")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 收集状态
                module_statuses = self.status_collector.collect_all_status()
                
                # 聚合数据
                aggregation_result = self.data_aggregator.aggregate_status(module_statuses)
                
                # 分析趋势
                self.status_analyzer.add_status_record(aggregation_result)
                
                # 检查预警
                new_alerts = self.alert_manager.check_alerts(aggregation_result)
                
                # 保存历史记录
                self.history_manager.save_status_record(aggregation_result, module_statuses)
                
                # 保存预警
                for alert in new_alerts:
                    self.history_manager.save_alert(alert)
                
                self.logger.debug("监控周期完成")
                
            except Exception as e:
                self.logger.error(f"监控过程中出错: {e}")
            
            # 等待下次监控
            time.sleep(self.monitor_interval)
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        module_statuses = self.status_collector.collect_all_status()
        return self.data_aggregator.aggregate_status(module_statuses)
    
    def get_trend_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """获取趋势分析"""
        return self.status_analyzer.analyze_trends(hours)
    
    def get_active_alerts(self) -> List[NetworkAlert]:
        """获取活跃预警"""
        return self.alert_manager.get_active_alerts()
    
    def get_alert_history(self, hours: int = 24) -> List[NetworkAlert]:
        """获取预警历史"""
        return self.alert_manager.get_alert_history(hours)
    
    def generate_report(self, include_trends: bool = True, include_alerts: bool = True) -> str:
        """生成综合报告"""
        # 获取当前状态
        current_status = self.get_current_status()
        
        # 获取趋势分析
        trend_analysis = {}
        if include_trends:
            trend_analysis = self.get_trend_analysis()
        
        # 获取活跃预警
        active_alerts = []
        if include_alerts:
            active_alerts = self.get_active_alerts()
        
        # 生成报告
        return self.report_generator.generate_status_report(
            current_status, trend_analysis, active_alerts
        )
    
    def export_report(self, filename: str = None, include_trends: bool = True, 
                     include_alerts: bool = True) -> str:
        """导出报告到文件"""
        report_content = self.generate_report(include_trends, include_alerts)
        return self.report_generator.export_report_to_file(report_content, filename)
    
    def create_dashboard(self, update_interval: int = 5000):
        """创建仪表板"""
        self.dashboard.create_dashboard(update_interval)
    
    def export_dashboard(self, filename: str = None) -> str:
        """导出仪表板截图"""
        return self.dashboard.export_dashboard(filename)
    
    def resolve_alert(self, alert_id: str):
        """解决预警"""
        self.alert_manager.resolve_alert(alert_id)
    
    def add_custom_alert_rule(self, rule_name: str, condition: Callable[[Dict[str, Any]], bool], 
                            severity: str, message_template: str):
        """添加自定义预警规则"""
        self.alert_manager.add_alert_rule(rule_name, condition, severity, message_template)
    
    def add_alert_callback(self, callback: Callable[[List[NetworkAlert]], None]):
        """添加预警回调函数"""
        self.alert_manager.add_alert_callback(callback)
    
    def get_status_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取状态历史"""
        return self.history_manager.get_status_history(hours)
    
    def get_modules_info(self) -> Dict[str, Dict[str, Any]]:
        """获取已注册模块信息"""
        return self.status_collector.modules.copy()
    
    def remove_module(self, module_name: str):
        """移除模块"""
        if module_name in self.status_collector.modules:
            del self.status_collector.modules[module_name]
            self.logger.info(f"已移除模块: {module_name}")
        else:
            raise ValueError(f"模块 {module_name} 不存在")
    
    def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        return {
            'timestamp': datetime.now(),
            'monitoring_active': self.monitoring,
            'registered_modules': len(self.status_collector.modules),
            'active_alerts': len(self.get_active_alerts()),
            'database_connected': True,  # 简化检查
            'components_status': {
                'status_collector': 'healthy',
                'data_aggregator': 'healthy',
                'status_analyzer': 'healthy',
                'alert_manager': 'healthy',
                'history_manager': 'healthy',
                'report_generator': 'healthy',
                'dashboard': 'healthy'
            }
        }


# 示例使用函数
def create_example_aggregator():
    """创建示例聚合器"""
    aggregator = NetworkStatusAggregator()
    
    # 注册示例模块
    aggregator.register_module(
        "google_dns", 
        "ping", 
        {"target": "8.8.8.8"}
    )
    
    aggregator.register_module(
        "httpbin", 
        "http", 
        {"url": "http://httpbin.org/get", "timeout": 5}
    )
    
    aggregator.register_module(
        "local_tcp", 
        "tcp", 
        {"host": "localhost", "port": 80}
    )
    
    aggregator.register_module(
        "system_status", 
        "system", 
        {}
    )
    
    return aggregator


if __name__ == "__main__":
    # 示例使用
    print("W9网络状态聚合器 - 示例运行")
    print("=" * 50)
    
    # 创建聚合器
    aggregator = create_example_aggregator()
    
    # 手动获取状态（不使用监控）
    print("获取当前状态...")
    current_status = aggregator.get_current_status()
    print(f"整体状态: {current_status['overall_status']}")
    print(f"健康度评分: {current_status['health_score']:.1f}")
    print(f"监控模块: {current_status['module_count']}个")
    
    # 生成报告
    print("\n生成综合报告...")
    report = aggregator.generate_report()
    print(report)
    
    print("\n示例运行完成！")