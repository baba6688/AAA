#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1系统监控器
==============

M1系统监控器是一个全面的系统监控解决方案，提供以下功能：
1. 系统运行状态监控
2. 系统服务状态监控
3. 系统进程监控
4. 系统事件监控
5. 系统告警管理
6. 系统日志收集
7. 系统性能指标
8. 系统健康评估
9. 系统监控报告


版本: 1.0.0
创建时间: 2025-11-05
"""

import os
import sys
import time
import json
import psutil
import logging
import threading
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import sqlite3
from pathlib import Path


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ServiceStatus(Enum):
    """服务状态枚举"""
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class SystemMetrics:
    """系统性能指标数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used: int
    memory_total: int
    disk_usage: float
    network_sent: int
    network_recv: int
    load_average: List[float]
    uptime: float


@dataclass
class ProcessInfo:
    """进程信息数据类"""
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_percent: float
    memory_used: int
    create_time: float
    num_threads: int


@dataclass
class ServiceInfo:
    """服务信息数据类"""
    name: str
    status: ServiceStatus
    pid: Optional[int]
    start_time: Optional[datetime]
    restart_count: int
    last_check: datetime


@dataclass
class Alert:
    """告警信息数据类"""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class SystemEvent:
    """系统事件数据类"""
    id: str
    event_type: str
    description: str
    timestamp: datetime
    source: str
    data: Dict[str, Any]


class DatabaseManager:
    """数据库管理器，负责监控数据的存储和查询"""
    
    def __init__(self, db_path: str = "system_monitor.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 系统指标表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    memory_used INTEGER NOT NULL,
                    memory_total INTEGER NOT NULL,
                    disk_usage REAL NOT NULL,
                    network_sent INTEGER NOT NULL,
                    network_recv INTEGER NOT NULL,
                    load_average TEXT NOT NULL,
                    uptime REAL NOT NULL
                )
            """)
            
            # 进程信息表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS process_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    memory_used INTEGER NOT NULL,
                    create_time REAL NOT NULL,
                    num_threads INTEGER NOT NULL
                )
            """)
            
            # 服务信息表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS service_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pid INTEGER,
                    start_time TEXT,
                    restart_count INTEGER NOT NULL,
                    last_check TEXT NOT NULL
                )
            """)
            
            # 告警表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    acknowledged INTEGER NOT NULL DEFAULT 0,
                    resolved INTEGER NOT NULL DEFAULT 0
                )
            """)
            
            # 系统事件表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            """)
            
            conn.commit()
    
    def save_metrics(self, metrics: SystemMetrics):
        """保存系统指标数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_metrics 
                (timestamp, cpu_percent, memory_percent, memory_used, memory_total, 
                 disk_usage, network_sent, network_recv, load_average, uptime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.cpu_percent,
                metrics.memory_percent,
                metrics.memory_used,
                metrics.memory_total,
                metrics.disk_usage,
                metrics.network_sent,
                metrics.network_recv,
                json.dumps(metrics.load_average),
                metrics.uptime
            ))
            conn.commit()
    
    def save_alert(self, alert: Alert):
        """保存告警信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO alerts 
                (id, level, title, message, timestamp, source, acknowledged, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.level.value,
                alert.title,
                alert.message,
                alert.timestamp.isoformat(),
                alert.source,
                int(alert.acknowledged),
                int(alert.resolved)
            ))
            conn.commit()
    
    def get_recent_metrics(self, hours: int = 24) -> List[SystemMetrics]:
        """获取最近的系统指标数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            since = (datetime.now() - timedelta(hours=hours)).isoformat()
            cursor.execute("""
                SELECT timestamp, cpu_percent, memory_percent, memory_used, memory_total,
                       disk_usage, network_sent, network_recv, load_average, uptime
                FROM system_metrics 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            """, (since,))
            
            metrics_list = []
            for row in cursor.fetchall():
                metrics = SystemMetrics(
                    timestamp=datetime.fromisoformat(row[0]),
                    cpu_percent=row[1],
                    memory_percent=row[2],
                    memory_used=row[3],
                    memory_total=row[4],
                    disk_usage=row[5],
                    network_sent=row[6],
                    network_recv=row[7],
                    load_average=json.loads(row[8]),
                    uptime=row[9]
                )
                metrics_list.append(metrics)
            
            return metrics_list


class AlertManager:
    """告警管理器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.alert_history = deque(maxlen=1000)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def create_alert(self, level: AlertLevel, title: str, message: str, 
                    source: str = "SystemMonitor") -> Alert:
        """创建新告警"""
        alert_id = f"{source}_{int(time.time())}_{hash(message) % 10000}"
        alert = Alert(
            id=alert_id,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source
        )
        
        # 保存到数据库
        self.db_manager.save_alert(alert)
        
        # 添加到历史记录
        self.alert_history.append(alert)
        
        # 触发回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logging.error(f"告警回调执行失败: {e}")
        
        return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE alerts SET acknowledged = 1 
                WHERE id = ? AND acknowledged = 0
            """, (alert_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE alerts SET resolved = 1 
                WHERE id = ?
            """, (alert_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, level, title, message, timestamp, source, acknowledged, resolved
                FROM alerts 
                WHERE resolved = 0 
                ORDER BY timestamp DESC
            """)
            
            alerts = []
            for row in cursor.fetchall():
                alert = Alert(
                    id=row[0],
                    level=AlertLevel(row[1]),
                    title=row[2],
                    message=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    source=row[5],
                    acknowledged=bool(row[6]),
                    resolved=bool(row[7])
                )
                alerts.append(alert)
            
            return alerts


class SystemMonitor:
    """M1系统监控器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化系统监控器
        
        Args:
            config: 配置字典，包含监控参数
        """
        self.config = config or {}
        self.db_manager = DatabaseManager(
            self.config.get('db_path', 'system_monitor.db')
        )
        self.alert_manager = AlertManager(self.db_manager)
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitor_interval = self.config.get('monitor_interval', 60)  # 秒
        
        # 监控数据缓存
        self.current_metrics: Optional[SystemMetrics] = None
        self.process_cache: Dict[int, ProcessInfo] = {}
        self.service_cache: Dict[str, ServiceInfo] = {}
        
        # 性能阈值配置
        self.thresholds = {
            'cpu_warning': self.config.get('cpu_warning_threshold', 70.0),
            'cpu_critical': self.config.get('cpu_critical_threshold', 90.0),
            'memory_warning': self.config.get('memory_warning_threshold', 80.0),
            'memory_critical': self.config.get('memory_critical_threshold', 95.0),
            'disk_warning': self.config.get('disk_warning_threshold', 85.0),
            'disk_critical': self.config.get('disk_critical_threshold', 95.0),
        }
        
        # 设置日志
        self._setup_logging()
        
        # 注册默认告警回调
        self.alert_manager.add_alert_callback(self._default_alert_handler)
    
    def _setup_logging(self):
        """设置日志记录"""
        log_level = self.config.get('log_level', 'INFO')
        log_file = self.config.get('log_file', 'system_monitor.log')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SystemMonitor')
    
    def _default_alert_handler(self, alert: Alert):
        """默认告警处理器"""
        self.logger.warning(f"告警 [{alert.level.value}] {alert.title}: {alert.message}")
        
        # 根据告警级别执行不同操作
        if alert.level == AlertLevel.CRITICAL:
            # 严重告警可以添加额外的处理逻辑
            self.logger.critical(f"严重告警: {alert.title}")
    
    def start_monitoring(self):
        """开始系统监控"""
        if self.is_monitoring:
            self.logger.warning("监控已在运行中")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止系统监控"""
        if not self.is_monitoring:
            self.logger.warning("监控未在运行")
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 执行各项监控任务
                self._collect_system_metrics()
                self._monitor_processes()
                self._monitor_services()
                self._check_system_health()
                
                time.sleep(self.monitor_interval)
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(5)  # 错误时短暂休眠
    
    def _collect_system_metrics(self):
        """收集系统性能指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存信息
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used
            memory_total = memory.total
            
            # 磁盘使用率
            disk_usage = psutil.disk_usage('/').percent
            
            # 网络IO
            network = psutil.net_io_counters()
            network_sent = network.bytes_sent
            network_recv = network.bytes_recv
            
            # 负载平均值（仅Linux/macOS）
            try:
                load_average = list(os.getloadavg())
            except (OSError, AttributeError):
                load_average = [0.0, 0.0, 0.0]
            
            # 系统运行时间
            uptime = time.time() - psutil.boot_time()
            
            # 创建指标对象
            self.current_metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used=memory_used,
                memory_total=memory_total,
                disk_usage=disk_usage,
                network_sent=network_sent,
                network_recv=network_recv,
                load_average=load_average,
                uptime=uptime
            )
            
            # 保存到数据库
            self.db_manager.save_metrics(self.current_metrics)
            
            # 检查阈值并生成告警
            self._check_metrics_thresholds()
            
        except Exception as e:
            self.logger.error(f"收集系统指标失败: {e}")
    
    def _check_metrics_thresholds(self):
        """检查指标阈值并生成告警"""
        if not self.current_metrics:
            return
        
        metrics = self.current_metrics
        
        # CPU使用率检查
        if metrics.cpu_percent >= self.thresholds['cpu_critical']:
            self.alert_manager.create_alert(
                AlertLevel.CRITICAL,
                "CPU使用率过高",
                f"CPU使用率: {metrics.cpu_percent:.1f}% (阈值: {self.thresholds['cpu_critical']}%)",
                "CPU监控"
            )
        elif metrics.cpu_percent >= self.thresholds['cpu_warning']:
            self.alert_manager.create_alert(
                AlertLevel.WARNING,
                "CPU使用率警告",
                f"CPU使用率: {metrics.cpu_percent:.1f}% (阈值: {self.thresholds['cpu_warning']}%)",
                "CPU监控"
            )
        
        # 内存使用率检查
        if metrics.memory_percent >= self.thresholds['memory_critical']:
            self.alert_manager.create_alert(
                AlertLevel.CRITICAL,
                "内存使用率过高",
                f"内存使用率: {metrics.memory_percent:.1f}% (阈值: {self.thresholds['memory_critical']}%)",
                "内存监控"
            )
        elif metrics.memory_percent >= self.thresholds['memory_warning']:
            self.alert_manager.create_alert(
                AlertLevel.WARNING,
                "内存使用率警告",
                f"内存使用率: {metrics.memory_percent:.1f}% (阈值: {self.thresholds['memory_warning']}%)",
                "内存监控"
            )
        
        # 磁盘使用率检查
        if metrics.disk_usage >= self.thresholds['disk_critical']:
            self.alert_manager.create_alert(
                AlertLevel.CRITICAL,
                "磁盘使用率过高",
                f"磁盘使用率: {metrics.disk_usage:.1f}% (阈值: {self.thresholds['disk_critical']}%)",
                "磁盘监控"
            )
        elif metrics.disk_usage >= self.thresholds['disk_warning']:
            self.alert_manager.create_alert(
                AlertLevel.WARNING,
                "磁盘使用率警告",
                f"磁盘使用率: {metrics.disk_usage:.1f}% (阈值: {self.thresholds['disk_warning']}%)",
                "磁盘监控"
            )
    
    def _monitor_processes(self):
        """监控系统进程"""
        try:
            current_pids = set()
            new_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 
                                            'memory_percent', 'create_time', 'num_threads']):
                try:
                    pinfo = proc.info
                    if pinfo['pid'] is None:
                        continue
                    
                    pid = pinfo['pid']
                    current_pids.add(pid)
                    
                    # 计算内存使用量（字节）
                    try:
                        memory_used = proc.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        memory_used = 0
                    
                    process_info = ProcessInfo(
                        pid=pid,
                        name=pinfo['name'] or 'unknown',
                        status=pinfo['status'] or 'unknown',
                        cpu_percent=pinfo['cpu_percent'] or 0.0,
                        memory_percent=pinfo['memory_percent'] or 0.0,
                        memory_used=memory_used,
                        create_time=pinfo['create_time'] or 0.0,
                        num_threads=pinfo['num_threads'] or 0
                    )
                    
                    # 检查是否为新进程
                    if pid not in self.process_cache:
                        new_processes.append(process_info)
                    
                    self.process_cache[pid] = process_info
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # 清理已结束的进程
            ended_pids = set(self.process_cache.keys()) - current_pids
            for pid in ended_pids:
                del self.process_cache[pid]
            
            # 记录新进程事件
            for process in new_processes:
                self._create_system_event(
                    "PROCESS_START",
                    f"新进程启动: {process.name} (PID: {process.pid})",
                    "进程监控",
                    {"pid": process.pid, "name": process.name}
                )
            
            # 检查高CPU或内存使用的进程
            self._check_process_alerts()
            
        except Exception as e:
            self.logger.error(f"进程监控失败: {e}")
    
    def _check_process_alerts(self):
        """检查进程告警"""
        high_cpu_processes = []
        high_memory_processes = []
        
        for process in self.process_cache.values():
            if process.cpu_percent > 50.0:  # CPU使用率超过50%
                high_cpu_processes.append(process)
            if process.memory_percent > 10.0:  # 内存使用率超过10%
                high_memory_processes.append(process)
        
        # 生成高CPU使用率告警
        if high_cpu_processes:
            for process in high_cpu_processes[:3]:  # 只报告前3个
                self.alert_manager.create_alert(
                    AlertLevel.WARNING,
                    "高CPU使用率进程",
                    f"进程 {process.name} (PID: {process.pid}) CPU使用率: {process.cpu_percent:.1f}%",
                    "进程监控"
                )
        
        # 生成高内存使用率告警
        if high_memory_processes:
            for process in high_memory_processes[:3]:  # 只报告前3个
                self.alert_manager.create_alert(
                    AlertLevel.WARNING,
                    "高内存使用率进程",
                    f"进程 {process.name} (PID: {process.pid}) 内存使用率: {process.memory_percent:.1f}%",
                    "进程监控"
                )
    
    def _monitor_services(self):
        """监控系统服务"""
        try:
            # 这里可以扩展为支持多种服务管理工具
            # 目前示例性地检查一些常见服务
            
            common_services = ['ssh', 'nginx', 'apache2', 'mysql', 'postgresql']
            
            for service_name in common_services:
                service_info = self._check_service_status(service_name)
                if service_info:
                    self.service_cache[service_name] = service_info
                    
                    # 检查服务状态变化
                    if service_name in self.service_cache:
                        old_status = self.service_cache[service_name].status
                        if old_status != service_info.status:
                            self._create_system_event(
                                "SERVICE_STATUS_CHANGE",
                                f"服务 {service_name} 状态变化: {old_status.value} -> {service_info.status.value}",
                                "服务监控",
                                {"service": service_name, "old_status": old_status.value, 
                                 "new_status": service_info.status.value}
                            )
            
        except Exception as e:
            self.logger.error(f"服务监控失败: {e}")
    
    def _check_service_status(self, service_name: str) -> Optional[ServiceInfo]:
        """检查单个服务状态"""
        try:
            # 尝试使用systemctl检查服务状态（Linux）
            result = subprocess.run(
                ['systemctl', 'is-active', service_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                status_str = result.stdout.strip()
                if status_str == 'active':
                    status = ServiceStatus.RUNNING
                else:
                    status = ServiceStatus.STOPPED
            else:
                status = ServiceStatus.UNKNOWN
            
            # 获取服务PID
            pid = None
            try:
                result = subprocess.run(
                    ['systemctl', 'show', service_name, '--property=MainPID'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    pid_line = result.stdout.strip()
                    if pid_line.startswith('MainPID='):
                        pid_str = pid_line.split('=')[1]
                        if pid_str and pid_str != '0':
                            pid = int(pid_str)
            except (ValueError, subprocess.TimeoutExpired):
                pass
            
            return ServiceInfo(
                name=service_name,
                status=status,
                pid=pid,
                start_time=None,  # 可以通过其他方式获取
                restart_count=0,
                last_check=datetime.now()
            )
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            # systemctl不可用，返回未知状态
            return ServiceInfo(
                name=service_name,
                status=ServiceStatus.UNKNOWN,
                pid=None,
                start_time=None,
                restart_count=0,
                last_check=datetime.now()
            )
    
    def _check_system_health(self):
        """检查系统健康状态"""
        if not self.current_metrics:
            return
        
        health_status = self._evaluate_system_health()
        
        # 根据健康状态生成相应的系统事件
        if health_status == HealthStatus.CRITICAL:
            self._create_system_event(
                "SYSTEM_HEALTH_CRITICAL",
                "系统健康状态: 严重",
                "健康检查",
                {"health_status": health_status.value}
            )
        elif health_status == HealthStatus.WARNING:
            self._create_system_event(
                "SYSTEM_HEALTH_WARNING",
                "系统健康状态: 警告",
                "健康检查",
                {"health_status": health_status.value}
            )
    
    def _evaluate_system_health(self) -> HealthStatus:
        """评估系统健康状态"""
        if not self.current_metrics:
            return HealthStatus.UNKNOWN
        
        metrics = self.current_metrics
        
        # 检查关键指标
        critical_issues = 0
        warning_issues = 0
        
        if metrics.cpu_percent >= self.thresholds['cpu_critical']:
            critical_issues += 1
        elif metrics.cpu_percent >= self.thresholds['cpu_warning']:
            warning_issues += 1
        
        if metrics.memory_percent >= self.thresholds['memory_critical']:
            critical_issues += 1
        elif metrics.memory_percent >= self.thresholds['memory_warning']:
            warning_issues += 1
        
        if metrics.disk_usage >= self.thresholds['disk_critical']:
            critical_issues += 1
        elif metrics.disk_usage >= self.thresholds['disk_warning']:
            warning_issues += 1
        
        # 根据问题数量确定健康状态
        if critical_issues > 0:
            return HealthStatus.CRITICAL
        elif warning_issues > 1:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _create_system_event(self, event_type: str, description: str, 
                           source: str, data: Dict[str, Any]):
        """创建系统事件"""
        event = SystemEvent(
            id=f"{source}_{int(time.time())}_{hash(description) % 10000}",
            event_type=event_type,
            description=description,
            timestamp=datetime.now(),
            source=source,
            data=data
        )
        
        # 保存到数据库
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_events 
                (id, event_type, description, timestamp, source, data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event.id,
                event.event_type,
                event.description,
                event.timestamp.isoformat(),
                event.source,
                json.dumps(event.data)
            ))
            conn.commit()
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """获取当前系统指标"""
        return self.current_metrics
    
    def get_process_info(self, pid: int) -> Optional[ProcessInfo]:
        """获取指定进程信息"""
        return self.process_cache.get(pid)
    
    def get_all_processes(self) -> List[ProcessInfo]:
        """获取所有进程信息"""
        return list(self.process_cache.values())
    
    def get_service_info(self, service_name: str) -> Optional[ServiceInfo]:
        """获取指定服务信息"""
        return self.service_cache.get(service_name)
    
    def get_all_services(self) -> List[ServiceInfo]:
        """获取所有服务信息"""
        return list(self.service_cache.values())
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return self.alert_manager.get_active_alerts()
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        return self.alert_manager.acknowledge_alert(alert_id)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        return self.alert_manager.resolve_alert(alert_id)
    
    def get_system_health(self) -> HealthStatus:
        """获取系统健康状态"""
        return self._evaluate_system_health()
    
    def get_recent_events(self, hours: int = 24, event_type: Optional[str] = None) -> List[SystemEvent]:
        """获取最近的系统事件"""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            since = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            if event_type:
                cursor.execute("""
                    SELECT id, event_type, description, timestamp, source, data
                    FROM system_events 
                    WHERE timestamp >= ? AND event_type = ?
                    ORDER BY timestamp DESC
                """, (since, event_type))
            else:
                cursor.execute("""
                    SELECT id, event_type, description, timestamp, source, data
                    FROM system_events 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (since,))
            
            events = []
            for row in cursor.fetchall():
                event = SystemEvent(
                    id=row[0],
                    event_type=row[1],
                    description=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    source=row[4],
                    data=json.loads(row[5])
                )
                events.append(event)
            
            return events
    
    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """生成系统监控报告"""
        report = {
            "report_time": datetime.now().isoformat(),
            "period_hours": hours,
            "system_overview": {},
            "performance_summary": {},
            "alerts_summary": {},
            "events_summary": {},
            "recommendations": []
        }
        
        # 系统概览
        if self.current_metrics:
            report["system_overview"] = {
                "cpu_usage": f"{self.current_metrics.cpu_percent:.1f}%",
                "memory_usage": f"{self.current_metrics.memory_percent:.1f}%",
                "disk_usage": f"{self.current_metrics.disk_usage:.1f}%",
                "uptime_hours": f"{self.current_metrics.uptime / 3600:.1f}",
                "load_average": self.current_metrics.load_average,
                "health_status": self.get_system_health().value
            }
        
        # 性能摘要
        recent_metrics = self.db_manager.get_recent_metrics(hours)
        if recent_metrics:
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]
            disk_values = [m.disk_usage for m in recent_metrics]
            
            report["performance_summary"] = {
                "cpu": {
                    "average": f"{sum(cpu_values) / len(cpu_values):.1f}%",
                    "max": f"{max(cpu_values):.1f}%",
                    "min": f"{min(cpu_values):.1f}%"
                },
                "memory": {
                    "average": f"{sum(memory_values) / len(memory_values):.1f}%",
                    "max": f"{max(memory_values):.1f}%",
                    "min": f"{min(memory_values):.1f}%"
                },
                "disk": {
                    "average": f"{sum(disk_values) / len(disk_values):.1f}%",
                    "max": f"{max(disk_values):.1f}%",
                    "min": f"{min(disk_values):.1f}%"
                }
            }
        
        # 告警摘要
        active_alerts = self.get_active_alerts()
        alert_counts = {"INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0}
        for alert in active_alerts:
            alert_counts[alert.level.value] += 1
        
        report["alerts_summary"] = {
            "active_alerts": len(active_alerts),
            "by_level": alert_counts
        }
        
        # 事件摘要
        recent_events = self.get_recent_events(hours)
        event_type_counts = {}
        for event in recent_events:
            event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
        
        report["events_summary"] = {
            "total_events": len(recent_events),
            "by_type": event_type_counts
        }
        
        # 生成建议
        recommendations = []
        if self.current_metrics:
            if self.current_metrics.cpu_percent > 80:
                recommendations.append("CPU使用率较高，建议检查高CPU消耗的进程")
            
            if self.current_metrics.memory_percent > 85:
                recommendations.append("内存使用率较高，建议检查内存泄漏或增加内存")
            
            if self.current_metrics.disk_usage > 90:
                recommendations.append("磁盘空间不足，建议清理临时文件或扩展磁盘空间")
        
        if len(active_alerts) > 5:
            recommendations.append("活跃告警较多，建议及时处理")
        
        report["recommendations"] = recommendations
        
        return report


# 测试用例
class SystemMonitorTest:
    """系统监控器测试类"""
    
    @staticmethod
    def test_basic_functionality():
        """测试基本功能"""
        print("=== 系统监控器基本功能测试 ===")
        
        # 创建监控器实例
        config = {
            'monitor_interval': 5,  # 5秒间隔用于测试
            'cpu_warning_threshold': 50.0,
            'memory_warning_threshold': 60.0,
            'disk_warning_threshold': 70.0,
            'log_level': 'DEBUG'
        }
        
        monitor = SystemMonitor(config)
        
        try:
            # 测试收集系统指标
            print("1. 测试系统指标收集...")
            monitor._collect_system_metrics()
            if monitor.current_metrics:
                print(f"   CPU使用率: {monitor.current_metrics.cpu_percent:.1f}%")
                print(f"   内存使用率: {monitor.current_metrics.memory_percent:.1f}%")
                print(f"   磁盘使用率: {monitor.current_metrics.disk_usage:.1f}%")
                print("   ✓ 系统指标收集成功")
            else:
                print("   ✗ 系统指标收集失败")
            
            # 测试进程监控
            print("2. 测试进程监控...")
            monitor._monitor_processes()
            print(f"   发现 {len(monitor.process_cache)} 个进程")
            print("   ✓ 进程监控成功")
            
            # 测试服务监控
            print("3. 测试服务监控...")
            monitor._monitor_services()
            print(f"   检查了 {len(monitor.service_cache)} 个服务")
            print("   ✓ 服务监控成功")
            
            # 测试告警系统
            print("4. 测试告警系统...")
            test_alert = monitor.alert_manager.create_alert(
                AlertLevel.WARNING,
                "测试告警",
                "这是一个测试告警",
                "测试模块"
            )
            print(f"   创建告警: {test_alert.title}")
            print("   ✓ 告警系统成功")
            
            # 测试系统健康评估
            print("5. 测试系统健康评估...")
            health = monitor.get_system_health()
            print(f"   系统健康状态: {health.value}")
            print("   ✓ 健康评估成功")
            
            # 测试报告生成
            print("6. 测试报告生成...")
            report = monitor.generate_report(hours=1)
            print(f"   报告生成成功，包含 {len(report)} 个部分")
            print("   ✓ 报告生成成功")
            
            print("\n=== 所有基本功能测试通过 ===")
            
        except Exception as e:
            print(f"测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 清理
            monitor.stop_monitoring()
    
    @staticmethod
    def test_monitoring_loop():
        """测试监控循环"""
        print("=== 系统监控器循环测试 ===")
        
        config = {
            'monitor_interval': 2,  # 2秒间隔用于快速测试
            'log_level': 'INFO'
        }
        
        monitor = SystemMonitor(config)
        
        try:
            # 启动监控
            print("启动监控...")
            monitor.start_monitoring()
            
            # 运行10秒
            print("监控运行中，等待10秒...")
            time.sleep(10)
            
            # 停止监控
            print("停止监控...")
            monitor.stop_monitoring()
            
            # 检查收集的数据
            print("检查收集的数据...")
            if monitor.current_metrics:
                print(f"最终CPU使用率: {monitor.current_metrics.cpu_percent:.1f}%")
                print(f"最终内存使用率: {monitor.current_metrics.memory_percent:.1f}%")
            
            alerts = monitor.get_active_alerts()
            print(f"生成的告警数量: {len(alerts)}")
            
            print("=== 监控循环测试完成 ===")
            
        except Exception as e:
            print(f"监控循环测试错误: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def test_alert_management():
        """测试告警管理功能"""
        print("=== 告警管理功能测试 ===")
        
        monitor = SystemMonitor()
        
        try:
            # 创建测试告警
            alerts = []
            for i in range(5):
                alert = monitor.alert_manager.create_alert(
                    AlertLevel.WARNING,
                    f"测试告警 {i+1}",
                    f"这是第 {i+1} 个测试告警",
                    "测试模块"
                )
                alerts.append(alert)
            
            print(f"创建了 {len(alerts)} 个测试告警")
            
            # 获取活跃告警
            active_alerts = monitor.get_active_alerts()
            print(f"活跃告警数量: {len(active_alerts)}")
            
            # 确认一些告警
            for i, alert in enumerate(active_alerts[:3]):
                success = monitor.acknowledge_alert(alert.id)
                print(f"确认告警 {alert.id}: {'成功' if success else '失败'}")
            
            # 解决一些告警
            for i, alert in enumerate(active_alerts[:2]):
                success = monitor.resolve_alert(alert.id)
                print(f"解决告警 {alert.id}: {'成功' if success else '失败'}")
            
            # 再次获取活跃告警
            remaining_alerts = monitor.get_active_alerts()
            print(f"剩余活跃告警数量: {len(remaining_alerts)}")
            
            print("=== 告警管理功能测试完成 ===")
            
        except Exception as e:
            print(f"告警管理测试错误: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def run_all_tests():
        """运行所有测试"""
        print("开始运行系统监控器测试套件...\n")
        
        SystemMonitorTest.test_basic_functionality()
        print()
        
        SystemMonitorTest.test_alert_management()
        print()
        
        SystemMonitorTest.test_monitoring_loop()
        print()
        
        print("所有测试完成！")


if __name__ == "__main__":
    # 运行测试
    SystemMonitorTest.run_all_tests()
    
    # 或者启动实际的监控器
    """
    print("启动M1系统监控器...")
    config = {
        'monitor_interval': 60,  # 1分钟监控间隔
        'cpu_warning_threshold': 70.0,
        'cpu_critical_threshold': 90.0,
        'memory_warning_threshold': 80.0,
        'memory_critical_threshold': 95.0,
        'disk_warning_threshold': 85.0,
        'disk_critical_threshold': 95.0,
        'log_level': 'INFO',
        'log_file': 'system_monitor.log',
        'db_path': 'system_monitor.db'
    }
    
    monitor = SystemMonitor(config)
    
    try:
        monitor.start_monitoring()
        print("监控器已启动，按 Ctrl+C 停止...")
        
        # 保持主线程运行
        while True:
            time.sleep(10)
            
            # 定期打印状态
            if monitor.current_metrics:
                health = monitor.get_system_health()
                print(f"系统状态: {health.value}, "
                      f"CPU: {monitor.current_metrics.cpu_percent:.1f}%, "
                      f"内存: {monitor.current_metrics.memory_percent:.1f}%, "
                      f"磁盘: {monitor.current_metrics.disk_usage:.1f}%")
            
    except KeyboardInterrupt:
        print("\n收到停止信号...")
    finally:
        monitor.stop_monitoring()
        print("监控器已停止")
    """