#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X7缓存监控器 - 主要实现
提供缓存状态监控、性能监控、告警管理等功能
"""

import time
import threading
import json
import logging
import sqlite3
import weakref

# 尝试导入psutil，如果失败则使用模拟模块
try:
    import psutil
except ImportError:
    from psutil import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import hashlib
import os


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CacheStatus(Enum):
    """缓存状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FULL = "full"
    ERROR = "error"


@dataclass
class CacheMetrics:
    """缓存指标数据"""
    timestamp: float
    hit_rate: float
    miss_rate: float
    hit_count: int
    miss_count: int
    size_bytes: int
    max_size_bytes: int
    item_count: int
    avg_access_time: float
    cpu_usage: float
    memory_usage: float
    status: str


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric: str
    condition: str  # '>', '<', '>=', '<=', '=='
    threshold: float
    level: AlertLevel
    enabled: bool = True
    cooldown: int = 300  # 5分钟冷却时间


@dataclass
class Alert:
    """告警信息"""
    id: str
    rule_name: str
    level: AlertLevel
    message: str
    timestamp: float
    value: float
    resolved: bool = False
    resolved_at: Optional[float] = None


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str = "cache_monitor.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建指标表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    hit_rate REAL,
                    miss_rate REAL,
                    hit_count INTEGER,
                    miss_count INTEGER,
                    size_bytes INTEGER,
                    max_size_bytes INTEGER,
                    item_count INTEGER,
                    avg_access_time REAL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    status TEXT
                )
            ''')
            
            # 创建告警表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    rule_name TEXT,
                    level TEXT,
                    message TEXT,
                    timestamp REAL,
                    value REAL,
                    resolved INTEGER DEFAULT 0,
                    resolved_at REAL
                )
            ''')
            
            # 创建告警规则表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alert_rules (
                    name TEXT PRIMARY KEY,
                    metric TEXT,
                    condition TEXT,
                    threshold REAL,
                    level TEXT,
                    enabled INTEGER DEFAULT 1,
                    cooldown INTEGER DEFAULT 300
                )
            ''')
            
            conn.commit()
    
    def save_metrics(self, metrics: CacheMetrics):
        """保存指标数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO metrics (
                    timestamp, hit_rate, miss_rate, hit_count, miss_count,
                    size_bytes, max_size_bytes, item_count, avg_access_time,
                    cpu_usage, memory_usage, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.hit_rate, metrics.miss_rate,
                metrics.hit_count, metrics.miss_count, metrics.size_bytes,
                metrics.max_size_bytes, metrics.item_count,
                metrics.avg_access_time, metrics.cpu_usage,
                metrics.memory_usage, metrics.status
            ))
            conn.commit()
    
    def get_metrics_history(self, hours: int = 24) -> List[CacheMetrics]:
        """获取历史指标数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            since = time.time() - (hours * 3600)
            cursor.execute('''
                SELECT * FROM metrics WHERE timestamp > ? ORDER BY timestamp
            ''', (since,))
            
            results = []
            for row in cursor.fetchall():
                results.append(CacheMetrics(*row[1:]))
            return results
    
    def save_alert(self, alert: Alert):
        """保存告警"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO alerts (
                    id, rule_name, level, message, timestamp, value,
                    resolved, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.rule_name, alert.level.value,
                alert.message, alert.timestamp, alert.value,
                int(alert.resolved), alert.resolved_at
            ))
            conn.commit()
    
    def get_alerts(self, hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            since = time.time() - (hours * 3600)
            cursor.execute('''
                SELECT * FROM alerts WHERE timestamp > ? ORDER BY timestamp DESC
            ''', (since,))
            
            results = []
            for row in cursor.fetchall():
                alert = Alert(
                    id=row[0], rule_name=row[1], level=AlertLevel(row[2]),
                    message=row[3], timestamp=row[4], value=row[5],
                    resolved=bool(row[6]), resolved_at=row[7]
                )
                results.append(alert)
            return results


class CacheMonitor:
    """X7缓存监控器主类"""
    
    def __init__(self, 
                 cache_backend=None,
                 max_size_mb: int = 100,
                 monitor_interval: int = 30,
                 db_path: str = "cache_monitor.db"):
        """
        初始化缓存监控器
        
        Args:
            cache_backend: 缓存后端对象
            max_size_mb: 最大缓存大小(MB)
            monitor_interval: 监控间隔(秒)
            db_path: 数据库文件路径
        """
        self.cache_backend = cache_backend
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.monitor_interval = monitor_interval
        self.db_manager = DatabaseManager(db_path)
        
        # 监控数据
        self.metrics_history = deque(maxlen=1000)
        self.hit_count = 0
        self.miss_count = 0
        self.access_times = deque(maxlen=100)
        self.current_size = 0
        self.item_count = 0
        
        # 告警管理
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.last_alert_time: Dict[str, float] = {}
        
        # 线程控制
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.RLock()
        
        # 性能统计
        self.start_time = time.time()
        self.total_requests = 0
        self.total_response_time = 0.0
        
        # 日志配置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 初始化默认告警规则
        self._init_default_alert_rules()
    
    def _init_default_alert_rules(self):
        """初始化默认告警规则"""
        default_rules = [
            AlertRule("缓存命中率过低", "hit_rate", "<", 0.8, AlertLevel.WARNING),
            AlertRule("缓存内存使用率过高", "memory_usage", ">", 0.9, AlertLevel.ERROR),
            AlertRule("缓存CPU使用率过高", "cpu_usage", ">", 0.8, AlertLevel.WARNING),
            AlertRule("缓存响应时间过长", "avg_access_time", ">", 0.1, AlertLevel.WARNING),
            AlertRule("缓存命中率严重过低", "hit_rate", "<", 0.5, AlertLevel.CRITICAL),
            AlertRule("缓存内存使用率严重过高", "memory_usage", ">", 0.95, AlertLevel.CRITICAL),
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
    
    def record_cache_access(self, hit: bool, response_time: float = 0.0, size: int = 0):
        """记录缓存访问"""
        with self._lock:
            if hit:
                self.hit_count += 1
            else:
                self.miss_count += 1
            
            self.total_requests += 1
            self.total_response_time += response_time
            self.access_times.append(response_time)
            
            if size > 0:
                self.current_size = size
                self.item_count += 1
    
    def get_current_metrics(self) -> CacheMetrics:
        """获取当前缓存指标"""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
            miss_rate = self.miss_count / total_requests if total_requests > 0 else 0.0
            
            avg_access_time = (
                sum(self.access_times) / len(self.access_times)
                if self.access_times else 0.0
            )
            
            # 系统资源使用情况
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent / 100.0
            
            # 缓存状态
            memory_usage_ratio = self.current_size / self.max_size_bytes
            if memory_usage_ratio >= 0.95:
                status = CacheStatus.FULL.value
            elif memory_usage_ratio >= 0.8:
                status = CacheStatus.ACTIVE.value
            else:
                status = CacheStatus.ACTIVE.value
            
            return CacheMetrics(
                timestamp=time.time(),
                hit_rate=hit_rate,
                miss_rate=miss_rate,
                hit_count=self.hit_count,
                miss_count=self.miss_count,
                size_bytes=self.current_size,
                max_size_bytes=self.max_size_bytes,
                item_count=self.item_count,
                avg_access_time=avg_access_time,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                status=status
            )
    
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        with self._lock:
            self.alert_rules[rule.name] = rule
            self.logger.info(f"添加告警规则: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """移除告警规则"""
        with self._lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                self.logger.info(f"移除告警规则: {rule_name}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def _check_alerts(self, metrics: CacheMetrics):
        """检查告警规则"""
        with self._lock:
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                # 检查冷却时间
                last_time = self.last_alert_time.get(rule.name, 0)
                if time.time() - last_time < rule.cooldown:
                    continue
                
                # 获取指标值
                value = getattr(metrics, rule.metric, None)
                if value is None:
                    continue
                
                # 检查条件
                triggered = False
                if rule.condition == ">" and value > rule.threshold:
                    triggered = True
                elif rule.condition == "<" and value < rule.threshold:
                    triggered = True
                elif rule.condition == ">=" and value >= rule.threshold:
                    triggered = True
                elif rule.condition == "<=" and value <= rule.threshold:
                    triggered = True
                elif rule.condition == "==" and value == rule.threshold:
                    triggered = True
                
                if triggered:
                    self._trigger_alert(rule, metrics, value)
    
    def _trigger_alert(self, rule: AlertRule, metrics: CacheMetrics, value: float):
        """触发告警"""
        alert_id = hashlib.md5(
            f"{rule.name}_{metrics.timestamp}".encode()
        ).hexdigest()
        
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            level=rule.level,
            message=f"{rule.name}: 当前值 {value:.4f}, 阈值 {rule.threshold}",
            timestamp=metrics.timestamp,
            value=value
        )
        
        self.active_alerts[alert_id] = alert
        self.last_alert_time[rule.name] = time.time()
        
        # 保存告警
        self.db_manager.save_alert(alert)
        
        # 调用回调函数
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"告警回调函数执行失败: {e}")
        
        self.logger.warning(f"触发告警: {alert.message}")
    
    def _monitor_loop(self):
        """监控循环"""
        self.logger.info("缓存监控器启动")
        
        while self._monitoring:
            try:
                # 获取当前指标
                metrics = self.get_current_metrics()
                
                # 保存指标
                self.db_manager.save_metrics(metrics)
                self.metrics_history.append(metrics)
                
                # 检查告警
                self._check_alerts(metrics)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(self.monitor_interval)
    
    def start_monitoring(self):
        """开始监控"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            self.logger.info("缓存监控器开始监控")
    
    def stop_monitoring(self):
        """停止监控"""
        if self._monitoring:
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join()
            self.logger.info("缓存监控器停止监控")
    
    def get_metrics_history(self, hours: int = 24) -> List[CacheMetrics]:
        """获取历史指标"""
        return self.db_manager.get_metrics_history(hours)
    
    def get_alerts(self, hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        return self.db_manager.get_alerts(hours)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        uptime = time.time() - self.start_time
        avg_response_time = (
            self.total_response_time / self.total_requests
            if self.total_requests > 0 else 0.0
        )
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / (self.hit_count + self.miss_count) 
                       if (self.hit_count + self.miss_count) > 0 else 0.0,
            "avg_response_time": avg_response_time,
            "current_size_mb": self.current_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "active_alerts": len(self.active_alerts)
        }
    
    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """生成监控报告"""
        metrics_history = self.get_metrics_history(hours)
        alerts = self.get_alerts(hours)
        performance = self.get_performance_summary()
        
        if not metrics_history:
            return {
                "period_hours": hours,
                "message": "没有可用的监控数据"
            }
        
        # 计算统计数据
        hit_rates = [m.hit_rate for m in metrics_history]
        response_times = [m.avg_access_time for m in metrics_history]
        cpu_usage = [m.cpu_usage for m in metrics_history]
        memory_usage = [m.memory_usage for m in metrics_history]
        
        return {
            "period_hours": hours,
            "generated_at": datetime.now().isoformat(),
            "performance": performance,
            "statistics": {
                "avg_hit_rate": sum(hit_rates) / len(hit_rates),
                "min_hit_rate": min(hit_rates),
                "max_hit_rate": max(hit_rates),
                "avg_response_time": sum(response_times) / len(response_times),
                "max_response_time": max(response_times),
                "avg_cpu_usage": sum(cpu_usage) / len(cpu_usage),
                "avg_memory_usage": sum(memory_usage) / len(memory_usage),
            },
            "alerts_summary": {
                "total_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a.level == AlertLevel.CRITICAL]),
                "error_alerts": len([a for a in alerts if a.level == AlertLevel.ERROR]),
                "warning_alerts": len([a for a in alerts if a.level == AlertLevel.WARNING]),
                "info_alerts": len([a for a in alerts if a.level == AlertLevel.INFO]),
            },
            "recommendations": self._generate_recommendations(metrics_history, alerts)
        }
    
    def _generate_recommendations(self, metrics_history: List[CacheMetrics], 
                                 alerts: List[Alert]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if not metrics_history:
            return ["建议增加监控时间以获得更准确的建议"]
        
        # 分析命中率
        hit_rates = [m.hit_rate for m in metrics_history]
        avg_hit_rate = sum(hit_rates) / len(hit_rates)
        
        if avg_hit_rate < 0.8:
            recommendations.append("缓存命中率较低，建议检查缓存策略和大小")
        
        if avg_hit_rate > 0.95:
            recommendations.append("缓存命中率很高，可以考虑适当减少缓存大小以节省内存")
        
        # 分析响应时间
        response_times = [m.avg_access_time for m in metrics_history]
        avg_response_time = sum(response_times) / len(response_times)
        
        if avg_response_time > 0.1:
            recommendations.append("平均响应时间较长，建议优化缓存算法或增加缓存大小")
        
        # 分析告警
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts:
            recommendations.append("存在严重告警，建议立即检查系统状态")
        
        # 分析内存使用
        memory_usage = [m.memory_usage for m in metrics_history]
        avg_memory = sum(memory_usage) / len(memory_usage)
        
        if avg_memory > 0.9:
            recommendations.append("系统内存使用率较高，建议优化内存使用或增加内存")
        
        if not recommendations:
            recommendations.append("系统运行正常，暂无优化建议")
        
        return recommendations
    
    def export_data(self, filepath: str, hours: int = 24):
        """导出监控数据"""
        report = self.generate_report(hours)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"监控数据已导出到: {filepath}")
    
    def optimize_monitoring(self):
        """优化监控性能"""
        with self._lock:
            # 清理过期的告警
            current_time = time.time()
            expired_alerts = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if current_time - alert.timestamp > 3600  # 1小时过期
            ]
            
            for alert_id in expired_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = current_time
                self.db_manager.save_alert(alert)
                del self.active_alerts[alert_id]
            
            # 清理历史数据
            if len(self.metrics_history) > self.metrics_history.maxlen:
                # 保留最新的数据
                recent_metrics = list(self.metrics_history)[-500:]
                self.metrics_history.clear()
                self.metrics_history.extend(recent_metrics)
            
            self.logger.info("监控性能优化完成")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()


# 装饰器：缓存监控装饰器
def monitor_cache_operation(monitor: CacheMonitor):
    """缓存操作监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                # 假设操作成功，记录为缓存命中
                response_time = time.time() - start_time
                monitor.record_cache_access(hit=True, response_time=response_time)
                return result
            except Exception as e:
                # 操作失败，记录为缓存未命中
                response_time = time.time() - start_time
                monitor.record_cache_access(hit=False, response_time=response_time)
                raise e
        
        return wrapper
    return decorator


# 示例缓存后端类
class SimpleCacheBackend:
    """简单的缓存后端示例"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        self.access_count += 1
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, size: int = 0):
        if len(self.cache) >= self.max_size:
            # 简单的LRU实现
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def delete(self, key: str):
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        self.cache.clear()
    
    def size(self) -> int:
        return len(self.cache)


if __name__ == "__main__":
    # 示例使用
    print("X7缓存监控器示例")
    
    # 创建缓存后端和监控器
    cache_backend = SimpleCacheBackend(max_size=50)
    monitor = CacheMonitor(
        cache_backend=cache_backend,
        max_size_mb=10,
        monitor_interval=5
    )
    
    # 添加告警回调
    def alert_handler(alert: Alert):
        print(f"告警通知: {alert.level.value} - {alert.message}")
    
    monitor.add_alert_callback(alert_handler)
    
    # 模拟缓存操作
    @monitor_cache_operation(monitor)
    def cache_operation(key: str):
        value = cache_backend.get(key)
        if value is None:
            # 模拟缓存未命中
            value = f"data_for_{key}"
            cache_backend.set(key, value, size=len(str(value)))
        return value
    
    # 开始监控
    with monitor:
        print("开始模拟缓存操作...")
        
        # 模拟一些缓存访问
        for i in range(20):
            key = f"key_{i % 10}"  # 重复访问一些键
            try:
                result = cache_operation(key)
                print(f"访问 {key}: {result}")
            except Exception as e:
                print(f"操作失败: {e}")
            
            time.sleep(0.1)
        
        # 等待一段时间让监控器收集数据
        time.sleep(10)
        
        # 生成报告
        report = monitor.generate_report(hours=1)
        print("\n=== 监控报告 ===")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        
        # 性能摘要
        summary = monitor.get_performance_summary()
        print("\n=== 性能摘要 ===")
        for key, value in summary.items():
            print(f"{key}: {value}")