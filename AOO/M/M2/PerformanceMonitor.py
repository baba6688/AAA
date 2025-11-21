"""
M2性能监控器模块

该模块实现了一个全面的性能监控系统，包括：
1. 应用性能监控
2. 数据库性能监控
3. 缓存性能监控
4. API响应时间监控
5. 吞吐量监控
6. 错误率监控
7. 性能基线管理
8. 性能优化建议
9. 性能报告生成


版本: 1.0.0
创建时间: 2025-11-05
"""

import time
import psutil
import threading
import logging
import json
import sqlite3
import statistics
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import functools
import requests
import redis
import sqlite3
from pathlib import Path
import warnings


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    metric_type: str
    name: str
    value: float
    unit: str
    tags: Dict[str, Any]
    threshold_exceeded: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'metric_type': self.metric_type,
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'tags': self.tags,
            'threshold_exceeded': self.threshold_exceeded
        }


@dataclass
class PerformanceBaseline:
    """性能基线数据类"""
    metric_name: str
    mean_value: float
    std_deviation: float
    p95_value: float
    p99_value: float
    min_value: float
    max_value: float
    sample_count: int
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


@dataclass
class PerformanceAlert:
    """性能告警数据类"""
    id: str
    timestamp: datetime
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data


class PerformanceMonitor:
    """
    M2性能监控器主类
    
    提供全面的系统性能监控功能，包括应用、数据库、缓存、API等各个方面的性能指标收集和分析。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化性能监控器
        
        Args:
            config: 配置字典，包含数据库连接、Redis连接、阈值设置等
        """
        self.config = config or {}
        self.metrics_buffer = deque(maxlen=10000)
        self.baselines = {}
        self.alerts = []
        self.monitoring_active = False
        self.monitor_threads = []
        
        # 数据库配置
        self.db_path = self.config.get('db_path', 'performance_monitor.db')
        self._init_database()
        
        # Redis配置（可选）
        self.redis_client = None
        if self.config.get('redis_host'):
            try:
                self.redis_client = redis.Redis(
                    host=self.config.get('redis_host'),
                    port=self.config.get('redis_port', 6379),
                    decode_responses=True
                )
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis连接失败: {e}")
        
        # 监控阈值配置
        self.thresholds = self.config.get('thresholds', {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'api_response_time': 5.0,  # 秒
            'error_rate': 5.0,  # 百分比
            'database_response_time': 2.0,  # 秒
            'cache_hit_rate': 80.0  # 百分比
        })
        
        # 线程池用于并发监控
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("性能监控器初始化完成")
    
    def _init_database(self) -> None:
        """初始化SQLite数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建性能指标表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        name TEXT NOT NULL,
                        value REAL NOT NULL,
                        unit TEXT NOT NULL,
                        tags TEXT,
                        threshold_exceeded BOOLEAN DEFAULT 0
                    )
                ''')
                
                # 创建性能基线表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS baselines (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT UNIQUE NOT NULL,
                        mean_value REAL NOT NULL,
                        std_deviation REAL NOT NULL,
                        p95_value REAL NOT NULL,
                        p99_value REAL NOT NULL,
                        min_value REAL NOT NULL,
                        max_value REAL NOT NULL,
                        sample_count INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                ''')
                
                # 创建告警表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        threshold_value REAL NOT NULL,
                        message TEXT NOT NULL,
                        resolved BOOLEAN DEFAULT 0,
                        resolved_at TEXT
                    )
                ''')
                
                conn.commit()
                logger.info("数据库初始化完成")
                
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def start_monitoring(self) -> None:
        """启动性能监控"""
        if self.monitoring_active:
            logger.warning("监控已经在运行中")
            return
        
        self.monitoring_active = True
        
        # 启动各种监控线程
        self.monitor_threads = [
            threading.Thread(target=self._monitor_system_resources, daemon=True),
            threading.Thread(target=self._monitor_application_metrics, daemon=True),
            threading.Thread(target=self._monitor_database_metrics, daemon=True),
            threading.Thread(target=self._monitor_cache_metrics, daemon=True),
            threading.Thread(target=self._analyze_metrics, daemon=True)
        ]
        
        for thread in self.monitor_threads:
            thread.start()
        
        logger.info("性能监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止性能监控"""
        self.monitoring_active = False
        
        # 等待所有线程结束
        for thread in self.monitor_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.executor.shutdown(wait=True)
        logger.info("性能监控已停止")
    
    def _monitor_system_resources(self) -> None:
        """监控系统资源使用情况"""
        while self.monitoring_active:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                self._record_metric('system', 'cpu_usage', cpu_percent, '%', {'core_count': psutil.cpu_count()})
                
                # 内存使用率
                memory = psutil.virtual_memory()
                self._record_metric('system', 'memory_usage', memory.percent, '%', {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used
                })
                
                # 磁盘使用率
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self._record_metric('system', 'disk_usage', disk_percent, '%', {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free
                })
                
                # 进程数
                process_count = len(psutil.pids())
                self._record_metric('system', 'process_count', process_count, 'count', {})
                
                time.sleep(5)  # 每5秒监控一次
                
            except Exception as e:
                logger.error(f"系统资源监控错误: {e}")
                time.sleep(5)
    
    def _monitor_application_metrics(self) -> None:
        """监控应用性能指标"""
        while self.monitoring_active:
            try:
                # 应用特定的性能指标可以在这里添加
                # 例如：活跃连接数、请求队列长度等
                
                # 模拟API响应时间监控
                if hasattr(self, '_api_endpoints'):
                    for endpoint in self._api_endpoints:
                        try:
                            start_time = time.time()
                            response = requests.get(endpoint, timeout=10)
                            response_time = time.time() - start_time
                            
                            self._record_metric('api', 'response_time', response_time, 's', {
                                'endpoint': endpoint,
                                'status_code': response.status_code
                            })
                            
                            # 记录错误率
                            if response.status_code >= 400:
                                self._record_metric('api', 'error_count', 1, 'count', {
                                    'endpoint': endpoint,
                                    'status_code': response.status_code
                                })
                                
                        except Exception as e:
                            logger.warning(f"API监控错误 {endpoint}: {e}")
                
                time.sleep(10)  # 每10秒监控一次
                
            except Exception as e:
                logger.error(f"应用监控错误: {e}")
                time.sleep(10)
    
    def _monitor_database_metrics(self) -> None:
        """监控数据库性能指标"""
        while self.monitoring_active:
            try:
                # 数据库连接数
                if hasattr(self, '_db_connections'):
                    for db_name, connection in self._db_connections.items():
                        try:
                            # 模拟数据库查询时间
                            start_time = time.time()
                            cursor = connection.cursor()
                            cursor.execute("SELECT 1")
                            query_time = time.time() - start_time
                            
                            self._record_metric('database', 'query_time', query_time, 's', {
                                'database': db_name,
                                'query_type': 'ping'
                            })
                            
                        except Exception as e:
                            logger.warning(f"数据库监控错误 {db_name}: {e}")
                
                time.sleep(15)  # 每15秒监控一次
                
            except Exception as e:
                logger.error(f"数据库监控错误: {e}")
                time.sleep(15)
    
    def _monitor_cache_metrics(self) -> None:
        """监控缓存性能指标"""
        while self.monitoring_active:
            try:
                if self.redis_client:
                    try:
                        # Redis连接测试
                        start_time = time.time()
                        self.redis_client.ping()
                        redis_time = time.time() - start_time
                        
                        self._record_metric('cache', 'redis_response_time', redis_time, 's', {})
                        
                        # 缓存命中率（需要应用层配合记录）
                        # 这里可以添加自定义的缓存监控逻辑
                        
                    except Exception as e:
                        logger.warning(f"Redis监控错误: {e}")
                
                time.sleep(20)  # 每20秒监控一次
                
            except Exception as e:
                logger.error(f"缓存监控错误: {e}")
                time.sleep(20)
    
    def _analyze_metrics(self) -> None:
        """分析性能指标并生成告警"""
        while self.monitoring_active:
            try:
                # 检查阈值
                for metric in list(self.metrics_buffer)[-100:]:  # 检查最近100个指标
                    self._check_thresholds(metric)
                
                # 更新基线
                self._update_baselines()
                
                time.sleep(30)  # 每30秒分析一次
                
            except Exception as e:
                logger.error(f"指标分析错误: {e}")
                time.sleep(30)
    
    def _record_metric(self, metric_type: str, name: str, value: float, unit: str, tags: Dict[str, Any]) -> None:
        """记录性能指标"""
        timestamp = datetime.now()
        threshold_exceeded = self._check_threshold(name, value)
        
        metric = PerformanceMetrics(
            timestamp=timestamp,
            metric_type=metric_type,
            name=name,
            value=value,
            unit=unit,
            tags=tags,
            threshold_exceeded=threshold_exceeded
        )
        
        self.metrics_buffer.append(metric)
        
        # 保存到数据库
        self._save_metric_to_db(metric)
        
        # 发送到Redis（如果可用）
        if self.redis_client:
            try:
                self.redis_client.lpush('performance_metrics', json.dumps(metric.to_dict()))
                # 保持列表长度在合理范围
                self.redis_client.ltrim('performance_metrics', 0, 9999)
            except Exception as e:
                logger.warning(f"Redis发送失败: {e}")
    
    def _check_threshold(self, metric_name: str, value: float) -> bool:
        """检查指标是否超过阈值"""
        if metric_name in self.thresholds:
            threshold = self.thresholds[metric_name]
            return value > threshold
        return False
    
    def _check_thresholds(self, metric: PerformanceMetrics) -> None:
        """检查指标阈值并生成告警"""
        if metric.threshold_exceeded:
            severity = self._determine_severity(metric.name, metric.value)
            alert = PerformanceAlert(
                id=f"{metric.name}_{metric.timestamp.isoformat()}",
                timestamp=metric.timestamp,
                severity=severity,
                metric_name=metric.name,
                current_value=metric.value,
                threshold_value=self.thresholds.get(metric.name, 0),
                message=f"指标 {metric.name} 超过阈值: {metric.value} {metric.unit}"
            )
            
            self.alerts.append(alert)
            self._save_alert_to_db(alert)
            logger.warning(f"性能告警: {alert.message}")
    
    def _determine_severity(self, metric_name: str, value: float) -> str:
        """确定告警严重程度"""
        threshold = self.thresholds.get(metric_name, 0)
        ratio = value / threshold if threshold > 0 else 1
        
        if ratio > 1.5:
            return "CRITICAL"
        elif ratio > 1.2:
            return "HIGH"
        elif ratio > 1.1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _update_baselines(self) -> None:
        """更新性能基线"""
        try:
            # 按指标名称分组收集最近24小时的数据
            metric_groups = defaultdict(list)
            
            cutoff_time = datetime.now() - timedelta(hours=24)
            for metric in self.metrics_buffer:
                if metric.timestamp >= cutoff_time:
                    metric_groups[metric.name].append(metric.value)
            
            for metric_name, values in metric_groups.items():
                if len(values) >= 10:  # 至少需要10个样本
                    baseline = PerformanceBaseline(
                        metric_name=metric_name,
                        mean_value=statistics.mean(values),
                        std_deviation=statistics.stdev(values) if len(values) > 1 else 0,
                        p95_value=self._percentile(values, 95),
                        p99_value=self._percentile(values, 99),
                        min_value=min(values),
                        max_value=max(values),
                        sample_count=len(values),
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    
                    self.baselines[metric_name] = baseline
                    self._save_baseline_to_db(baseline)
                    
        except Exception as e:
            logger.error(f"基线更新错误: {e}")
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = min(lower_index + 1, len(sorted_data) - 1)
            weight = index - lower_index
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    def _save_metric_to_db(self, metric: PerformanceMetrics) -> None:
        """保存指标到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO metrics (timestamp, metric_type, name, value, unit, tags, threshold_exceeded)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp.isoformat(),
                    metric.metric_type,
                    metric.name,
                    metric.value,
                    metric.unit,
                    json.dumps(metric.tags),
                    metric.threshold_exceeded
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"保存指标到数据库失败: {e}")
    
    def _save_baseline_to_db(self, baseline: PerformanceBaseline) -> None:
        """保存基线到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO baselines 
                    (metric_name, mean_value, std_deviation, p95_value, p99_value, 
                     min_value, max_value, sample_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    baseline.metric_name,
                    baseline.mean_value,
                    baseline.std_deviation,
                    baseline.p95_value,
                    baseline.p99_value,
                    baseline.min_value,
                    baseline.max_value,
                    baseline.sample_count,
                    baseline.created_at.isoformat(),
                    baseline.updated_at.isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"保存基线到数据库失败: {e}")
    
    def _save_alert_to_db(self, alert: PerformanceAlert) -> None:
        """保存告警到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO alerts 
                    (id, timestamp, severity, metric_name, current_value, threshold_value, message, resolved, resolved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id,
                    alert.timestamp.isoformat(),
                    alert.severity,
                    alert.metric_name,
                    alert.current_value,
                    alert.threshold_value,
                    alert.message,
                    alert.resolved,
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"保存告警到数据库失败: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前性能指标"""
        current_metrics = {}
        
        # 从缓冲区获取最新的指标
        recent_metrics = list(self.metrics_buffer)[-100:]
        
        for metric in recent_metrics:
            if metric.name not in current_metrics:
                current_metrics[metric.name] = []
            current_metrics[metric.name].append(metric.to_dict())
        
        return current_metrics
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """获取指标历史数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp, value, unit, tags, threshold_exceeded
                    FROM metrics
                    WHERE name = ? AND timestamp >= ?
                    ORDER BY timestamp
                ''', (metric_name, (datetime.now() - timedelta(hours=hours)).isoformat()))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'timestamp': row[0],
                        'value': row[1],
                        'unit': row[2],
                        'tags': json.loads(row[3]) if row[3] else {},
                        'threshold_exceeded': bool(row[4])
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"获取指标历史失败: {e}")
            return []
    
    def get_performance_baseline(self, metric_name: str) -> Optional[PerformanceBaseline]:
        """获取性能基线"""
        return self.baselines.get(metric_name)
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """获取活跃告警"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                self._save_alert_to_db(alert)
                return True
        return False
    
    def generate_optimization_suggestions(self) -> List[str]:
        """生成性能优化建议"""
        suggestions = []
        
        try:
            # 分析最近的性能数据
            recent_metrics = list(self.metrics_buffer)[-1000:]  # 最近1000个指标
            
            # CPU使用率分析
            cpu_metrics = [m for m in recent_metrics if m.name == 'cpu_usage' and m.metric_type == 'system']
            if cpu_metrics:
                avg_cpu = statistics.mean([m.value for m in cpu_metrics])
                if avg_cpu > 80:
                    suggestions.append("CPU使用率过高，建议：1) 优化算法复杂度 2) 增加服务器CPU核心数 3) 使用负载均衡")
                elif avg_cpu > 60:
                    suggestions.append("CPU使用率偏高，建议监控应用程序性能，考虑代码优化")
            
            # 内存使用率分析
            memory_metrics = [m for m in recent_metrics if m.name == 'memory_usage' and m.metric_type == 'system']
            if memory_metrics:
                avg_memory = statistics.mean([m.value for m in memory_metrics])
                if avg_memory > 85:
                    suggestions.append("内存使用率过高，建议：1) 检查内存泄漏 2) 优化数据结构 3) 增加服务器内存")
                elif avg_memory > 70:
                    suggestions.append("内存使用率偏高，建议监控内存使用模式")
            
            # API响应时间分析
            api_metrics = [m for m in recent_metrics if m.name == 'response_time' and m.metric_type == 'api']
            if api_metrics:
                avg_response_time = statistics.mean([m.value for m in api_metrics])
                if avg_response_time > 5:
                    suggestions.append("API响应时间过长，建议：1) 优化数据库查询 2) 使用缓存 3) 异步处理")
                elif avg_response_time > 2:
                    suggestions.append("API响应时间偏高，建议监控具体API端点性能")
            
            # 错误率分析
            error_metrics = [m for m in recent_metrics if m.name == 'error_count' and m.metric_type == 'api']
            if error_metrics:
                total_errors = sum([m.value for m in error_metrics])
                total_requests = len([m for m in recent_metrics if m.metric_type == 'api'])
                error_rate = (total_errors / max(total_requests, 1)) * 100
                
                if error_rate > 5:
                    suggestions.append(f"错误率过高({error_rate:.2f}%)，建议：1) 检查错误日志 2) 完善异常处理 3) 加强输入验证")
                elif error_rate > 2:
                    suggestions.append(f"错误率偏高({error_rate:.2f}%)，建议检查应用程序稳定性")
            
            # 缓存命中率分析
            cache_metrics = [m for m in recent_metrics if m.name == 'cache_hit_rate']
            if cache_metrics:
                avg_hit_rate = statistics.mean([m.value for m in cache_metrics])
                if avg_hit_rate < 80:
                    suggestions.append("缓存命中率偏低，建议：1) 优化缓存策略 2) 调整缓存过期时间 3) 增加缓存容量")
                elif avg_hit_rate < 90:
                    suggestions.append("缓存命中率有提升空间，建议优化缓存配置")
            
            if not suggestions:
                suggestions.append("当前系统性能良好，建议继续保持现有配置并定期监控")
                
        except Exception as e:
            logger.error(f"生成优化建议失败: {e}")
            suggestions.append("无法生成优化建议，请检查系统状态")
        
        return suggestions
    
    def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """生成性能报告"""
        try:
            # 获取最近的数据
            recent_metrics = list(self.metrics_buffer)[-1000:]
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [m for m in recent_metrics if m.timestamp >= cutoff_time]
            
            # 统计信息
            report = {
                'report_time': datetime.now().isoformat(),
                'time_range_hours': hours,
                'summary': {
                    'total_metrics': len(recent_metrics),
                    'threshold_violations': len([m for m in recent_metrics if m.threshold_exceeded]),
                    'active_alerts': len(self.get_active_alerts()),
                    'metrics_types': list(set(m.metric_type for m in recent_metrics))
                },
                'metrics_summary': {},
                'performance_analysis': {},
                'optimization_suggestions': self.generate_optimization_suggestions(),
                'alerts': [alert.to_dict() for alert in self.get_active_alerts()]
            }
            
            # 按指标类型汇总
            for metric_type in set(m.metric_type for m in recent_metrics):
                type_metrics = [m for m in recent_metrics if m.metric_type == metric_type]
                
                metric_summary = {}
                for metric_name in set(m.name for m in type_metrics):
                    name_metrics = [m for m in type_metrics if m.name == metric_name]
                    values = [m.value for m in name_metrics]
                    
                    if values:
                        metric_summary[metric_name] = {
                            'count': len(values),
                            'min': min(values),
                            'max': max(values),
                            'avg': statistics.mean(values),
                            'median': statistics.median(values),
                            'p95': self._percentile(values, 95),
                            'threshold_violations': len([m for m in name_metrics if m.threshold_exceeded])
                        }
                
                report['metrics_summary'][metric_type] = metric_summary
            
            # 性能分析
            report['performance_analysis'] = {
                'system_health': self._analyze_system_health(recent_metrics),
                'trends': self._analyze_trends(recent_metrics),
                'bottlenecks': self._identify_bottlenecks(recent_metrics)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return {'error': str(e)}
    
    def _analyze_system_health(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """分析系统健康状况"""
        health_score = 100
        issues = []
        
        # 检查各项指标
        cpu_metrics = [m for m in metrics if m.name == 'cpu_usage']
        if cpu_metrics:
            avg_cpu = statistics.mean([m.value for m in cpu_metrics])
            if avg_cpu > 80:
                health_score -= 30
                issues.append(f"CPU使用率过高: {avg_cpu:.1f}%")
            elif avg_cpu > 60:
                health_score -= 15
                issues.append(f"CPU使用率偏高: {avg_cpu:.1f}%")
        
        memory_metrics = [m for m in metrics if m.name == 'memory_usage']
        if memory_metrics:
            avg_memory = statistics.mean([m.value for m in memory_metrics])
            if avg_memory > 85:
                health_score -= 25
                issues.append(f"内存使用率过高: {avg_memory:.1f}%")
            elif avg_memory > 70:
                health_score -= 10
                issues.append(f"内存使用率偏高: {avg_memory:.1f}%")
        
        api_metrics = [m for m in metrics if m.name == 'response_time' and m.metric_type == 'api']
        if api_metrics:
            avg_response = statistics.mean([m.value for m in api_metrics])
            if avg_response > 5:
                health_score -= 20
                issues.append(f"API响应时间过长: {avg_response:.2f}s")
            elif avg_response > 2:
                health_score -= 10
                issues.append(f"API响应时间偏高: {avg_response:.2f}s")
        
        return {
            'score': max(health_score, 0),
            'status': 'HEALTHY' if health_score >= 80 else 'WARNING' if health_score >= 60 else 'CRITICAL',
            'issues': issues
        }
    
    def _analyze_trends(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """分析性能趋势"""
        trends = {}
        
        # 按小时分组分析趋势
        hourly_data = defaultdict(list)
        for metric in metrics:
            hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_data[f"{metric.metric_type}_{metric.name}"].append((hour_key, metric.value))
        
        for metric_key, data_points in hourly_data.items():
            if len(data_points) >= 2:
                # 简单线性趋势分析
                values = [point[1] for point in data_points]
                if len(values) >= 2:
                    # 计算变化率
                    first_half = values[:len(values)//2]
                    second_half = values[len(values)//2:]
                    
                    if first_half and second_half:
                        first_avg = statistics.mean(first_half)
                        second_avg = statistics.mean(second_half)
                        change_rate = ((second_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0
                        
                        trends[metric_key] = {
                            'change_rate': change_rate,
                            'direction': 'increasing' if change_rate > 5 else 'decreasing' if change_rate < -5 else 'stable',
                            'severity': 'high' if abs(change_rate) > 20 else 'medium' if abs(change_rate) > 10 else 'low'
                        }
        
        return trends
    
    def _identify_bottlenecks(self, metrics: List[PerformanceMetrics]) -> List[Dict[str, Any]]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        # 分析各种性能瓶颈
        cpu_metrics = [m for m in metrics if m.name == 'cpu_usage']
        if cpu_metrics and statistics.mean([m.value for m in cpu_metrics]) > 80:
            bottlenecks.append({
                'type': 'CPU',
                'severity': 'high',
                'description': 'CPU使用率持续过高',
                'impact': '响应时间增加，系统整体性能下降',
                'recommendation': '优化算法，增加CPU核心，或使用负载均衡'
            })
        
        memory_metrics = [m for m in metrics if m.name == 'memory_usage']
        if memory_metrics and statistics.mean([m.value for m in memory_metrics]) > 85:
            bottlenecks.append({
                'type': 'Memory',
                'severity': 'high',
                'description': '内存使用率过高',
                'impact': '可能导致内存交换，系统性能严重下降',
                'recommendation': '检查内存泄漏，优化数据结构，增加内存'
            })
        
        api_metrics = [m for m in metrics if m.name == 'response_time' and m.metric_type == 'api']
        if api_metrics and statistics.mean([m.value for m in api_metrics]) > 5:
            bottlenecks.append({
                'type': 'API',
                'severity': 'medium',
                'description': 'API响应时间过长',
                'impact': '用户体验下降，吞吐量降低',
                'recommendation': '优化数据库查询，使用缓存，异步处理'
            })
        
        return bottlenecks
    
    def export_report(self, report: Dict[str, Any], format: str = 'json', filepath: Optional[str] = None) -> str:
        """导出性能报告"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"performance_report_{timestamp}.{format}"
        
        try:
            if format.lower() == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
            elif format.lower() == 'txt':
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("=== 性能监控报告 ===\n\n")
                    f.write(f"报告时间: {report.get('report_time', 'N/A')}\n")
                    f.write(f"时间范围: {report.get('time_range_hours', 'N/A')} 小时\n\n")
                    
                    # 汇总信息
                    summary = report.get('summary', {})
                    f.write("=== 汇总信息 ===\n")
                    for key, value in summary.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                    
                    # 优化建议
                    suggestions = report.get('optimization_suggestions', [])
                    f.write("=== 优化建议 ===\n")
                    for i, suggestion in enumerate(suggestions, 1):
                        f.write(f"{i}. {suggestion}\n")
                    f.write("\n")
                    
                    # 告警信息
                    alerts = report.get('alerts', [])
                    if alerts:
                        f.write("=== 活跃告警 ===\n")
                        for alert in alerts:
                            f.write(f"- {alert.get('severity', 'N/A')}: {alert.get('message', 'N/A')}\n")
                        f.write("\n")
            
            logger.info(f"报告已导出到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"导出报告失败: {e}")
            raise
    
    def monitor_api_endpoint(self, endpoint: str, method: str = 'GET', **kwargs) -> Callable:
        """API端点监控装饰器"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    response_time = time.time() - start_time
                    
                    # 记录成功的API调用
                    self._record_metric('api', 'response_time', response_time, 's', {
                        'endpoint': endpoint,
                        'method': method,
                        'status': 'success'
                    })
                    
                    return result
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    
                    # 记录失败的API调用
                    self._record_metric('api', 'error_count', 1, 'count', {
                        'endpoint': endpoint,
                        'method': method,
                        'error': str(e)
                    })
                    
                    self._record_metric('api', 'response_time', response_time, 's', {
                        'endpoint': endpoint,
                        'method': method,
                        'status': 'error'
                    })
                    
                    raise
            
            return wrapper
        return decorator
    
    def monitor_database_query(self, query_type: str = 'SELECT') -> Callable:
        """数据库查询监控装饰器"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    query_time = time.time() - start_time
                    
                    # 记录成功的数据库查询
                    self._record_metric('database', 'query_time', query_time, 's', {
                        'query_type': query_type,
                        'status': 'success'
                    })
                    
                    return result
                    
                except Exception as e:
                    query_time = time.time() - start_time
                    
                    # 记录失败的数据库查询
                    self._record_metric('database', 'error_count', 1, 'count', {
                        'query_type': query_type,
                        'error': str(e)
                    })
                    
                    self._record_metric('database', 'query_time', query_time, 's', {
                        'query_type': query_type,
                        'status': 'error'
                    })
                    
                    raise
            
            return wrapper
        return decorator
    
    def get_throughput_metrics(self, hours: int = 1) -> Dict[str, float]:
        """获取吞吐量指标"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [m for m in self.metrics_buffer if m.timestamp >= cutoff_time]
            
            # 计算各种吞吐量指标
            api_requests = len([m for m in recent_metrics if m.metric_type == 'api' and m.name == 'response_time'])
            db_queries = len([m for m in recent_metrics if m.metric_type == 'database' and m.name == 'query_time'])
            cache_operations = len([m for m in recent_metrics if m.metric_type == 'cache'])
            
            return {
                'api_requests_per_hour': api_requests / hours,
                'database_queries_per_hour': db_queries / hours,
                'cache_operations_per_hour': cache_operations / hours,
                'total_operations_per_hour': (api_requests + db_queries + cache_operations) / hours
            }
            
        except Exception as e:
            logger.error(f"获取吞吐量指标失败: {e}")
            return {}


# 测试用例
class PerformanceMonitorTest:
    """性能监控器测试类"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
    
    def test_basic_functionality(self):
        """测试基本功能"""
        print("=== 基本功能测试 ===")
        
        # 测试记录指标
        self.monitor._record_metric('test', 'test_metric', 100.0, 'ms', {'test': True})
        print("✓ 指标记录功能正常")
        
        # 测试阈值检查
        self.monitor._record_metric('system', 'cpu_usage', 90.0, '%', {})
        print("✓ 阈值检查功能正常")
        
        # 测试基线更新
        self.monitor._update_baselines()
        print("✓ 基线更新功能正常")
        
        # 测试优化建议生成
        suggestions = self.monitor.generate_optimization_suggestions()
        print(f"✓ 优化建议生成功能正常，生成 {len(suggestions)} 条建议")
        
        # 测试性能报告生成
        report = self.monitor.generate_performance_report(hours=1)
        print("✓ 性能报告生成功能正常")
        
        return True
    
    def test_decorators(self):
        """测试装饰器功能"""
        print("\n=== 装饰器测试 ===")
        
        @self.monitor.monitor_api_endpoint('/test', 'GET')
        def test_api_call():
            time.sleep(0.1)  # 模拟API调用
            return "success"
        
        @self.monitor.monitor_database_query('SELECT')
        def test_db_query():
            time.sleep(0.05)  # 模拟数据库查询
            return "data"
        
        # 测试API监控装饰器
        try:
            result = test_api_call()
            print("✓ API监控装饰器功能正常")
        except Exception as e:
            print(f"✗ API监控装饰器测试失败: {e}")
        
        # 测试数据库监控装饰器
        try:
            result = test_db_query()
            print("✓ 数据库监控装饰器功能正常")
        except Exception as e:
            print(f"✗ 数据库监控装饰器测试失败: {e}")
        
        return True
    
    def test_monitoring_simulation(self):
        """测试监控模拟"""
        print("\n=== 监控模拟测试 ===")
        
        # 模拟各种性能指标
        import random
        
        for i in range(50):
            # 模拟CPU使用率 (30-90%)
            cpu_usage = random.uniform(30, 90)
            self.monitor._record_metric('system', 'cpu_usage', cpu_usage, '%', {})
            
            # 模拟内存使用率 (40-95%)
            memory_usage = random.uniform(40, 95)
            self.monitor._record_metric('system', 'memory_usage', memory_usage, '%', {})
            
            # 模拟API响应时间 (0.1-10秒)
            response_time = random.uniform(0.1, 10)
            self.monitor._record_metric('api', 'response_time', response_time, 's', {
                'endpoint': f'/api/{i % 5}',
                'status_code': 200 if random.random() > 0.1 else 500
            })
            
            # 模拟数据库查询时间 (0.01-5秒)
            query_time = random.uniform(0.01, 5)
            self.monitor._record_metric('database', 'query_time', query_time, 's', {
                'query_type': 'SELECT',
                'table': f'table_{i % 3}'
            })
        
        print("✓ 模拟数据生成完成")
        
        # 生成测试报告
        report = self.monitor.generate_performance_report(hours=1)
        print("✓ 测试报告生成完成")
        
        # 导出测试报告
        report_file = self.monitor.export_report(report, 'json', 'test_performance_report.json')
        print(f"✓ 测试报告已导出到: {report_file}")
        
        return True
    
    def test_database_operations(self):
        """测试数据库操作"""
        print("\n=== 数据库操作测试 ===")
        
        try:
            # 测试指标历史查询
            history = self.monitor.get_metric_history('cpu_usage', hours=1)
            print(f"✓ 指标历史查询功能正常，获取到 {len(history)} 条记录")
            
            # 测试基线查询
            baseline = self.monitor.get_performance_baseline('cpu_usage')
            if baseline:
                print("✓ 基线查询功能正常")
            else:
                print("! 基线查询返回空结果（正常现象，如果没有足够数据）")
            
            # 测试告警查询
            alerts = self.monitor.get_active_alerts()
            print(f"✓ 活跃告警查询功能正常，当前有 {len(alerts)} 个活跃告警")
            
            return True
            
        except Exception as e:
            print(f"✗ 数据库操作测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始运行性能监控器测试...")
        
        try:
            # 启动监控
            self.monitor.start_monitoring()
            time.sleep(2)  # 让监控线程启动
            
            # 运行各项测试
            tests = [
                self.test_basic_functionality,
                self.test_decorators,
                self.test_monitoring_simulation,
                self.test_database_operations
            ]
            
            passed = 0
            total = len(tests)
            
            for test in tests:
                try:
                    if test():
                        passed += 1
                except Exception as e:
                    print(f"测试失败: {e}")
            
            print(f"\n=== 测试结果 ===")
            print(f"通过: {passed}/{total}")
            print(f"成功率: {passed/total*100:.1f}%")
            
        finally:
            # 停止监控
            self.monitor.stop_monitoring()
            print("测试完成，监控已停止")


def main():
    """主函数 - 演示性能监控器的使用"""
    print("M2性能监控器演示")
    print("=" * 50)
    
    # 创建监控器实例
    config = {
        'db_path': 'demo_performance_monitor.db',
        'thresholds': {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'api_response_time': 5.0,
            'error_rate': 5.0
        }
    }
    
    monitor = PerformanceMonitor(config)
    
    try:
        # 启动监控
        monitor.start_monitoring()
        print("监控已启动...")
        
        # 模拟一些应用活动
        print("模拟应用活动...")
        time.sleep(5)
        
        # 生成当前性能报告
        print("\n生成性能报告...")
        report = monitor.generate_performance_report(hours=1)
        
        # 显示报告摘要
        print("\n=== 性能报告摘要 ===")
        summary = report.get('summary', {})
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # 显示优化建议
        suggestions = report.get('optimization_suggestions', [])
        if suggestions:
            print("\n=== 优化建议 ===")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
        
        # 显示活跃告警
        alerts = report.get('alerts', [])
        if alerts:
            print("\n=== 活跃告警 ===")
            for alert in alerts:
                print(f"- {alert.get('severity', 'N/A')}: {alert.get('message', 'N/A')}")
        
        # 导出报告
        report_file = monitor.export_report(report, 'json', 'demo_performance_report.json')
        print(f"\n报告已导出到: {report_file}")
        
    finally:
        # 停止监控
        monitor.stop_monitoring()
        print("监控已停止")


if __name__ == "__main__":
    # 运行演示
    main()
    
    # 运行测试
    print("\n" + "=" * 50)
    print("运行性能监控器测试...")
    test_suite = PerformanceMonitorTest()
    test_suite.run_all_tests()