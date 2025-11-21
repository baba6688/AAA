#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y6存储监控器
实现存储状态监控、性能监控、告警管理等功能
"""

import os
import sys
import time
import json
import logging
import threading
import psutil
import sqlite3
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import schedule
import yaml
import subprocess
import shutil
from collections import defaultdict, deque


@dataclass
class StorageInfo:
    """存储信息数据类"""
    mount_point: str
    total_space: int
    used_space: int
    free_space: int
    usage_percent: float
    io_read_bytes: int
    io_write_bytes: int
    io_read_count: int
    io_write_count: int
    timestamp: str


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    mount_point: str
    read_speed: float  # MB/s
    write_speed: float  # MB/s
    io_utilization: float  # %
    avg_queue_depth: float
    timestamp: str


@dataclass
class Alert:
    """告警信息数据类"""
    id: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    title: str
    message: str
    mount_point: str
    timestamp: str
    resolved: bool = False


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 存储信息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS storage_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mount_point TEXT NOT NULL,
                    total_space INTEGER NOT NULL,
                    used_space INTEGER NOT NULL,
                    free_space INTEGER NOT NULL,
                    usage_percent REAL NOT NULL,
                    io_read_bytes INTEGER NOT NULL,
                    io_write_bytes INTEGER NOT NULL,
                    io_read_count INTEGER NOT NULL,
                    io_write_count INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            # 性能指标表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mount_point TEXT NOT NULL,
                    read_speed REAL NOT NULL,
                    write_speed REAL NOT NULL,
                    io_utilization REAL NOT NULL,
                    avg_queue_depth REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            # 告警表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    mount_point TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # 监控配置表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monitor_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT UNIQUE NOT NULL,
                    config_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
    
    def insert_storage_info(self, storage_info: StorageInfo):
        """插入存储信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO storage_info 
                (mount_point, total_space, used_space, free_space, usage_percent,
                 io_read_bytes, io_write_bytes, io_read_count, io_write_count, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                storage_info.mount_point, storage_info.total_space,
                storage_info.used_space, storage_info.free_space,
                storage_info.usage_percent, storage_info.io_read_bytes,
                storage_info.io_write_bytes, storage_info.io_read_count,
                storage_info.io_write_count, storage_info.timestamp
            ))
            conn.commit()
    
    def insert_performance_metrics(self, metrics: PerformanceMetrics):
        """插入性能指标"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO performance_metrics 
                (mount_point, read_speed, write_speed, io_utilization, avg_queue_depth, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                metrics.mount_point, metrics.read_speed, metrics.write_speed,
                metrics.io_utilization, metrics.avg_queue_depth, metrics.timestamp
            ))
            conn.commit()
    
    def insert_alert(self, alert: Alert):
        """插入告警"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO alerts 
                (id, level, title, message, mount_point, timestamp, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.level, alert.title, alert.message,
                alert.mount_point, alert.timestamp, alert.resolved
            ))
            conn.commit()
    
    def get_storage_info_history(self, mount_point: str, hours: int = 24) -> List[StorageInfo]:
        """获取存储信息历史"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            since = (datetime.now() - timedelta(hours=hours)).isoformat()
            cursor.execute('''
                SELECT mount_point, total_space, used_space, free_space, usage_percent,
                       io_read_bytes, io_write_bytes, io_read_count, io_write_count, timestamp
                FROM storage_info 
                WHERE mount_point = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (mount_point, since))
            
            results = []
            for row in cursor.fetchall():
                results.append(StorageInfo(*row))
            return results
    
    def get_performance_metrics_history(self, mount_point: str, hours: int = 24) -> List[PerformanceMetrics]:
        """获取性能指标历史"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            since = (datetime.now() - timedelta(hours=hours)).isoformat()
            cursor.execute('''
                SELECT mount_point, read_speed, write_speed, io_utilization, avg_queue_depth, timestamp
                FROM performance_metrics 
                WHERE mount_point = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (mount_point, since))
            
            results = []
            for row in cursor.fetchall():
                results.append(PerformanceMetrics(*row))
            return results
    
    def get_alerts(self, limit: int = 100) -> List[Alert]:
        """获取告警"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, level, title, message, mount_point, timestamp, resolved
                FROM alerts 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append(Alert(*row))
            return results
    
    def cleanup_old_data(self, days: int = 30):
        """清理旧数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute('DELETE FROM storage_info WHERE timestamp < ?', (cutoff,))
            cursor.execute('DELETE FROM performance_metrics WHERE timestamp < ?', (cutoff,))
            cursor.execute('DELETE FROM alerts WHERE timestamp < ? AND resolved = TRUE', (cutoff,))
            
            conn.commit()


class AlertManager:
    """告警管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.alert_cooldown = {}  # 告警冷却时间
        self.cooldown_minutes = config.get('alert_cooldown_minutes', 5)
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)
    
    def send_alert(self, alert: Alert):
        """发送告警"""
        # 检查冷却时间
        alert_key = f"{alert.mount_point}_{alert.title}"
        now = time.time()
        
        if alert_key in self.alert_cooldown:
            if now - self.alert_cooldown[alert_key] < self.cooldown_minutes * 60:
                return  # 仍在冷却时间内
        
        # 更新冷却时间
        self.alert_cooldown[alert_key] = now
        
        # 发送告警
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"告警处理器执行失败: {e}")
    
    def check_thresholds(self, storage_info: StorageInfo, config: Dict[str, Any]) -> List[Alert]:
        """检查阈值并生成告警"""
        alerts = []
        mount_point = storage_info.mount_point
        
        # 存储使用率告警
        usage_thresholds = config.get('usage_thresholds', {})
        if storage_info.usage_percent >= usage_thresholds.get('critical', 95):
            alerts.append(Alert(
                id=f"{mount_point}_usage_critical_{int(time.time())}",
                level="CRITICAL",
                title="存储使用率严重告警",
                message=f"{mount_point} 存储使用率已达到 {storage_info.usage_percent:.1f}%",
                mount_point=mount_point,
                timestamp=datetime.now().isoformat()
            ))
        elif storage_info.usage_percent >= usage_thresholds.get('warning', 85):
            alerts.append(Alert(
                id=f"{mount_point}_usage_warning_{int(time.time())}",
                level="WARNING",
                title="存储使用率告警",
                message=f"{mount_point} 存储使用率已达到 {storage_info.usage_percent:.1f}%",
                mount_point=mount_point,
                timestamp=datetime.now().isoformat()
            ))
        
        # 剩余空间告警
        free_thresholds = config.get('free_space_thresholds', {})
        free_gb = storage_info.free_space / (1024**3)
        if free_gb <= free_thresholds.get('critical', 1):
            alerts.append(Alert(
                id=f"{mount_point}_free_critical_{int(time.time())}",
                level="CRITICAL",
                title="存储空间严重不足",
                message=f"{mount_point} 剩余空间仅剩 {free_gb:.2f} GB",
                mount_point=mount_point,
                timestamp=datetime.now().isoformat()
            ))
        elif free_gb <= free_thresholds.get('warning', 5):
            alerts.append(Alert(
                id=f"{mount_point}_free_warning_{int(time.time())}",
                level="WARNING",
                title="存储空间不足",
                message=f"{mount_point} 剩余空间仅剩 {free_gb:.2f} GB",
                mount_point=mount_point,
                timestamp=datetime.now().isoformat()
            ))
        
        return alerts


class EmailAlertHandler:
    """邮件告警处理器"""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
    
    def __call__(self, alert: Alert):
        """发送邮件告警"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = ', '.join(self.smtp_config['to_emails'])
            msg['Subject'] = f"[{alert.level}] Y6存储监控器 - {alert.title}"
            
            body = f"""
            告警详情：
            
            告警级别：{alert.level}
            挂载点：{alert.mount_point}
            标题：{alert.title}
            消息：{alert.message}
            时间：{alert.timestamp}
            
            请及时处理相关问题。
            
            Y6存储监控器
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(self.smtp_config['smtp_server'], self.smtp_config['smtp_port'])
            if self.smtp_config.get('use_tls', True):
                server.starttls()
            if self.smtp_config.get('username') and self.smtp_config.get('password'):
                server.login(self.smtp_config['username'], self.smtp_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            logging.info(f"邮件告警已发送: {alert.title}")
        except Exception as e:
            logging.error(f"发送邮件告警失败: {e}")


class LogAlertHandler:
    """日志告警处理器"""
    
    def __call__(self, alert: Alert):
        """记录告警日志"""
        log_message = f"[{alert.level}] {alert.mount_point}: {alert.title} - {alert.message}"
        
        if alert.level == "CRITICAL":
            logging.critical(log_message)
        elif alert.level == "ERROR":
            logging.error(log_message)
        elif alert.level == "WARNING":
            logging.warning(log_message)
        else:
            logging.info(log_message)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.prev_io_stats = {}
        self.prev_time = None
    
    def get_disk_performance(self, mount_point: str) -> PerformanceMetrics:
        """获取磁盘性能指标"""
        try:
            # 获取当前时间
            current_time = time.time()
            
            # 获取磁盘IO统计
            io_stats = psutil.disk_io_counters(perdisk=True)
            
            # 查找对应挂载点的磁盘
            disk_name = None
            for disk, stats in io_stats.items():
                try:
                    # 尝试获取磁盘的挂载点
                    partitions = psutil.disk_partitions()
                    for partition in partitions:
                        if partition.device == disk and mount_point in partition.mountpoint:
                            disk_name = disk
                            break
                    if disk_name:
                        break
                except:
                    continue
            
            if not disk_name or disk_name not in io_stats:
                # 如果找不到特定磁盘，使用总计统计
                total_stats = psutil.disk_io_counters()
                if not total_stats:
                    return self._default_metrics(mount_point)
                
                current_read_bytes = total_stats.read_bytes
                current_write_bytes = total_stats.write_bytes
                current_read_count = total_stats.read_count
                current_write_count = total_stats.write_count
            else:
                stats = io_stats[disk_name]
                current_read_bytes = stats.read_bytes
                current_write_bytes = stats.write_bytes
                current_read_count = stats.read_count
                current_write_count = stats.write_count
            
            # 计算性能指标
            if self.prev_time and disk_name in self.prev_io_stats:
                time_diff = current_time - self.prev_time
                if time_diff > 0:
                    read_speed = (current_read_bytes - self.prev_io_stats[disk_name]['read_bytes']) / time_diff / (1024 * 1024)
                    write_speed = (current_write_bytes - self.prev_io_stats[disk_name]['write_bytes']) / time_diff / (1024 * 1024)
                else:
                    read_speed = write_speed = 0.0
            else:
                read_speed = write_speed = 0.0
            
            # 模拟IO利用率和队列深度
            io_utilization = min(100.0, (read_speed + write_speed) / 100)  # 简单估算
            avg_queue_depth = io_utilization / 10  # 简单估算
            
            # 更新历史数据
            self.prev_io_stats[disk_name] = {
                'read_bytes': current_read_bytes,
                'write_bytes': current_write_bytes,
                'read_count': current_read_count,
                'write_count': current_read_count
            }
            self.prev_time = current_time
            
            return PerformanceMetrics(
                mount_point=mount_point,
                read_speed=max(0, read_speed),
                write_speed=max(0, write_speed),
                io_utilization=max(0, min(100, io_utilization)),
                avg_queue_depth=max(0, avg_queue_depth),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logging.error(f"获取磁盘性能失败: {e}")
            return self._default_metrics(mount_point)
    
    def _default_metrics(self, mount_point: str) -> PerformanceMetrics:
        """默认性能指标"""
        return PerformanceMetrics(
            mount_point=mount_point,
            read_speed=0.0,
            write_speed=0.0,
            io_utilization=0.0,
            avg_queue_depth=0.0,
            timestamp=datetime.now().isoformat()
        )


class StorageMonitor:
    """存储监控器主类"""
    
    def __init__(self, config_path: str = "storage_monitor_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.db_manager = DatabaseManager(self.config.get('database_path', 'storage_monitor.db'))
        self.alert_manager = AlertManager(self.config)
        self.performance_monitor = PerformanceMonitor()
        
        # 设置告警处理器
        self._setup_alert_handlers()
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitored_mounts = []
        
        # 统计数据
        self.stats = {
            'total_monitors': 0,
            'alerts_generated': 0,
            'data_points_collected': 0
        }
        
        # 设置日志
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            'monitor_interval': 60,  # 监控间隔（秒）
            'database_path': 'storage_monitor.db',
            'log_level': 'INFO',
            'log_file': 'storage_monitor.log',
            'alert_cooldown_minutes': 5,
            'usage_thresholds': {
                'warning': 85,
                'critical': 95
            },
            'free_space_thresholds': {
                'warning': 5,  # GB
                'critical': 1   # GB
            },
            'smtp_config': {
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'use_tls': True,
                'from_email': 'monitor@example.com',
                'to_emails': ['admin@example.com'],
                'username': '',
                'password': ''
            },
            'monitored_mounts': ['/'],
            'cleanup_old_data_days': 30,
            'report_generation_time': '09:00'
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    # 合并配置
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            else:
                # 创建默认配置文件
                self._save_config(default_config)
                return default_config
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
            return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """保存配置文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logging.error(f"保存配置文件失败: {e}")
    
    def _setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        log_file = self.config.get('log_file', 'storage_monitor.log')
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def _setup_alert_handlers(self):
        """设置告警处理器"""
        # 日志处理器
        self.alert_manager.add_handler(LogAlertHandler())
        
        # 邮件处理器（如果配置了邮件）
        smtp_config = self.config.get('smtp_config', {})
        if smtp_config.get('to_emails'):
            self.alert_manager.add_handler(EmailAlertHandler(smtp_config))
    
    def get_storage_info(self, mount_point: str) -> Optional[StorageInfo]:
        """获取存储信息"""
        try:
            # 获取磁盘使用情况
            usage = psutil.disk_usage(mount_point)
            
            # 获取磁盘IO统计
            io_counters = psutil.disk_io_counters(perdisk=True)
            total_read_bytes = 0
            total_write_bytes = 0
            total_read_count = 0
            total_write_count = 0
            
            # 汇总所有磁盘的IO统计
            for disk, stats in io_counters.items():
                try:
                    # 检查磁盘是否与挂载点相关
                    partitions = psutil.disk_partitions()
                    for partition in partitions:
                        if partition.device == disk and mount_point in partition.mountpoint:
                            total_read_bytes += stats.read_bytes
                            total_write_bytes += stats.write_bytes
                            total_read_count += stats.read_count
                            total_write_count += stats.write_count
                            break
                except:
                    continue
            
            # 如果没有找到特定磁盘，使用总计统计
            if total_read_bytes == 0:
                total_io = psutil.disk_io_counters()
                if total_io:
                    total_read_bytes = total_io.read_bytes
                    total_write_bytes = total_io.write_bytes
                    total_read_count = total_io.read_count
                    total_write_count = total_io.write_count
            
            return StorageInfo(
                mount_point=mount_point,
                total_space=usage.total,
                used_space=usage.used,
                free_space=usage.free,
                usage_percent=(usage.used / usage.total) * 100,
                io_read_bytes=total_read_bytes,
                io_write_bytes=total_write_bytes,
                io_read_count=total_read_count,
                io_write_count=total_write_count,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logging.error(f"获取存储信息失败 {mount_point}: {e}")
            return None
    
    def monitor_storage(self):
        """执行存储监控"""
        logging.info("开始存储监控...")
        
        for mount_point in self.monitored_mounts:
            try:
                # 获取存储信息
                storage_info = self.get_storage_info(mount_point)
                if storage_info:
                    # 保存到数据库
                    self.db_manager.insert_storage_info(storage_info)
                    self.stats['data_points_collected'] += 1
                    
                    # 获取性能指标
                    performance_metrics = self.performance_monitor.get_disk_performance(mount_point)
                    self.db_manager.insert_performance_metrics(performance_metrics)
                    
                    # 检查告警
                    alerts = self.alert_manager.check_thresholds(storage_info, self.config)
                    for alert in alerts:
                        self.db_manager.insert_alert(alert)
                        self.alert_manager.send_alert(alert)
                        self.stats['alerts_generated'] += 1
                    
                    logging.debug(f"监控完成 {mount_point}: 使用率 {storage_info.usage_percent:.1f}%")
                
            except Exception as e:
                logging.error(f"监控存储失败 {mount_point}: {e}")
        
        # 清理旧数据
        if self.stats['total_monitors'] % 100 == 0:  # 每100次监控清理一次
            self.db_manager.cleanup_old_data(self.config.get('cleanup_old_data_days', 30))
        
        self.stats['total_monitors'] += 1
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            logging.warning("监控已在运行中")
            return
        
        self.monitored_mounts = self.config.get('monitored_mounts', ['/'])
        self.is_monitoring = True
        
        # 在后台线程中运行监控
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logging.info("存储监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logging.info("存储监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        interval = self.config.get('monitor_interval', 60)
        
        while self.is_monitoring:
            try:
                self.monitor_storage()
                time.sleep(interval)
            except Exception as e:
                logging.error(f"监控循环异常: {e}")
                time.sleep(5)  # 异常时短暂等待
    
    def get_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            'is_monitoring': self.is_monitoring,
            'monitored_mounts': self.monitored_mounts,
            'stats': self.stats.copy(),
            'config': self.config.copy()
        }
    
    def get_storage_report(self, mount_point: str, hours: int = 24) -> Dict[str, Any]:
        """生成存储报告"""
        try:
            # 获取历史数据
            storage_history = self.db_manager.get_storage_info_history(mount_point, hours)
            performance_history = self.db_manager.get_performance_metrics_history(mount_point, hours)
            
            if not storage_history:
                return {'error': '没有找到监控数据'}
            
            # 计算统计数据
            usage_values = [s.usage_percent for s in storage_history]
            read_speeds = [p.read_speed for p in performance_history]
            write_speeds = [p.write_speed for p in performance_history]
            
            report = {
                'mount_point': mount_point,
                'report_period_hours': hours,
                'generated_at': datetime.now().isoformat(),
                'storage_stats': {
                    'avg_usage_percent': sum(usage_values) / len(usage_values),
                    'max_usage_percent': max(usage_values),
                    'min_usage_percent': min(usage_values),
                    'current_usage_percent': storage_history[0].usage_percent,
                    'total_space_gb': storage_history[0].total_space / (1024**3),
                    'used_space_gb': storage_history[0].used_space / (1024**3),
                    'free_space_gb': storage_history[0].free_space / (1024**3)
                },
                'performance_stats': {
                    'avg_read_speed_mbps': sum(read_speeds) / len(read_speeds) if read_speeds else 0,
                    'avg_write_speed_mbps': sum(write_speeds) / len(write_speeds) if write_speeds else 0,
                    'max_read_speed_mbps': max(read_speeds) if read_speeds else 0,
                    'max_write_speed_mbps': max(write_speeds) if write_speeds else 0
                },
                'data_points': len(storage_history)
            }
            
            return report
            
        except Exception as e:
            logging.error(f"生成存储报告失败: {e}")
            return {'error': str(e)}
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        # 验证配置
        required_keys = ['monitor_interval', 'usage_thresholds', 'free_space_thresholds']
        for key in required_keys:
            if key not in new_config:
                raise ValueError(f"缺少必需的配置项: {key}")
        
        # 更新配置
        self.config.update(new_config)
        self.config['updated_at'] = datetime.now().isoformat()
        
        # 保存配置
        self._save_config(self.config)
        
        logging.info("配置已更新")
    
    def get_alerts(self, limit: int = 100) -> List[Alert]:
        """获取告警列表"""
        return self.db_manager.get_alerts(limit)
    
    def optimize_performance(self):
        """性能优化"""
        try:
            # 清理数据库碎片
            with self.db_manager.db_manager.db_path as conn:
                conn.execute('VACUUM')
            
            # 清理旧数据
            self.db_manager.cleanup_old_data(self.config.get('cleanup_old_data_days', 30))
            
            # 优化监控间隔
            current_interval = self.config.get('monitor_interval', 60)
            if current_interval < 30:
                self.config['monitor_interval'] = 30
                self._save_config(self.config)
                logging.info("监控间隔已优化为30秒")
            
            logging.info("性能优化完成")
            
        except Exception as e:
            logging.error(f"性能优化失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Y6存储监控器')
    parser.add_argument('--config', default='storage_monitor_config.yaml', help='配置文件路径')
    parser.add_argument('--daemon', action='store_true', help='以守护进程模式运行')
    parser.add_argument('--status', action='store_true', help='显示监控状态')
    parser.add_argument('--report', help='生成存储报告（指定挂载点）')
    parser.add_argument('--hours', type=int, default=24, help='报告时间范围（小时）')
    
    args = parser.parse_args()
    
    # 创建监控器
    monitor = StorageMonitor(args.config)
    
    if args.status:
        status = monitor.get_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return
    
    if args.report:
        report = monitor.get_storage_report(args.report, args.hours)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    
    if args.daemon:
        try:
            monitor.start_monitoring()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
    else:
        # 单次监控
        monitor.monitor_storage()
        print("监控完成")


if __name__ == "__main__":
    main()