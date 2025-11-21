#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M8健康检查器 - 全面的系统健康监控和评估工具

该模块实现了一个完整的健康检查器类，支持多维度的系统健康监控，
包括系统资源、服务状态、数据库连接、网络连通性、组件状态等。

功能特性:
- 系统健康检查 (CPU、内存、磁盘使用率等)
- 服务健康检查 (应用程序服务状态监控)
- 数据库健康检查 (连接状态、查询性能等)
- 网络健康检查 (连通性、延迟、带宽等)
- 组件健康检查 (系统组件状态评估)
- 健康状态综合评估
- 健康报告生成 (JSON、文本格式)
- 健康预警机制 (阈值监控、告警通知)
- 健康检查调度 (定时检查、异步执行)


创建时间: 2025-11-05
版本: 1.0.0
"""

import asyncio
import json
import logging
import psutil
import sqlite3
import socket
import subprocess
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"          # 健康
    WARNING = "warning"          # 警告
    CRITICAL = "critical"        # 严重
    UNKNOWN = "unknown"          # 未知


class CheckType(Enum):
    """检查类型枚举"""
    SYSTEM = "system"            # 系统检查
    SERVICE = "service"          # 服务检查
    DATABASE = "database"        # 数据库检查
    NETWORK = "network"          # 网络检查
    COMPONENT = "component"      # 组件检查


@dataclass
class HealthMetric:
    """健康指标数据类"""
    name: str                    # 指标名称
    value: float                 # 指标值
    unit: str                    # 单位
    status: HealthStatus         # 健康状态
    threshold_warning: float     # 警告阈值
    threshold_critical: float    # 严重阈值
    timestamp: datetime          # 检查时间
    description: str = ""        # 描述信息


@dataclass
class HealthCheckResult:
    """健康检查结果数据类"""
    check_type: CheckType        # 检查类型
    component_name: str          # 组件名称
    status: HealthStatus         # 整体状态
    metrics: List[HealthMetric]  # 指标列表
    timestamp: datetime          # 检查时间
    duration: float              # 检查耗时(秒)
    message: str = ""            # 附加信息


@dataclass
class HealthReport:
    """健康报告数据类"""
    overall_status: HealthStatus     # 整体健康状态
    timestamp: datetime              # 报告生成时间
    checks_performed: int            # 执行的检查数量
    healthy_components: int          # 健康组件数量
    warning_components: int          # 警告组件数量
    critical_components: int         # 严重组件数量
    results: List[HealthCheckResult] # 检查结果列表
    recommendations: List[str]       # 建议列表


class AlertManager:
    """告警管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化告警管理器
        
        Args:
            config: 告警配置信息
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.alert_history = []
        
    def send_alert(self, message: str, severity: HealthStatus, 
                   component: str = "") -> None:
        """
        发送告警通知
        
        Args:
            message: 告警消息
            severity: 严重程度
            component: 组件名称
        """
        alert = {
            "timestamp": datetime.now(),
            "message": message,
            "severity": severity.value,
            "component": component
        }
        
        self.alert_history.append(alert)
        
        # 记录日志
        log_level = {
            HealthStatus.CRITICAL: logging.ERROR,
            HealthStatus.WARNING: logging.WARNING,
            HealthStatus.HEALTHY: logging.INFO
        }.get(severity, logging.INFO)
        
        self.logger.log(log_level, f"[{component}] {message}")
        
        # 发送邮件通知 (如果配置了)
        if self.config.get("email", {}).get("enabled", False):
            self._send_email_alert(alert)
    
    def _send_email_alert(self, alert: Dict[str, Any]) -> None:
        """发送邮件告警"""
        try:
            email_config = self.config["email"]
            msg = MIMEMultipart()
            msg['From'] = email_config["from"]
            msg['To'] = email_config["to"]
            msg['Subject'] = f"系统告警: {alert['severity'].upper()}"
            
            body = f"""
            系统健康检查告警通知
            
            时间: {alert['timestamp']}
            组件: {alert['component']}
            严重程度: {alert['severity'].upper()}
            消息: {alert['message']}
            
            请及时处理相关问题。
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            server.starttls()
            server.login(email_config["username"], email_config["password"])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"发送邮件告警失败: {e}")


class HealthChecker:
    """健康检查器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化健康检查器
        
        Args:
            config: 配置信息
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        self.alert_manager = AlertManager(self.config.get("alerts", {}))
        self.check_history = []
        self.running_checks = {}
        self._setup_logging()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "thresholds": {
                "cpu_warning": 70.0,
                "cpu_critical": 90.0,
                "memory_warning": 80.0,
                "memory_critical": 95.0,
                "disk_warning": 80.0,
                "disk_critical": 90.0,
                "network_latency_warning": 100.0,
                "network_latency_critical": 500.0,
                "service_response_time_warning": 5.0,
                "service_response_time_critical": 10.0,
                "db_connection_timeout_warning": 5.0,
                "db_connection_timeout_critical": 10.0
            },
            "check_intervals": {
                "system": 60,        # 系统检查间隔(秒)
                "service": 30,       # 服务检查间隔(秒)
                "database": 120,     # 数据库检查间隔(秒)
                "network": 30,       # 网络检查间隔(秒)
                "component": 300     # 组件检查间隔(秒)
            },
            "services": [
                {"name": "web_server", "host": "localhost", "port": 80, "path": "/"},
                {"name": "api_server", "host": "localhost", "port": 8080, "path": "/health"}
            ],
            "databases": [
                {"name": "main_db", "type": "sqlite", "path": "health_check.db"},
                {"name": "analytics_db", "type": "sqlite", "path": "analytics.db"}
            ],
            "network_targets": [
                {"name": "google_dns", "host": "8.8.8.8", "port": 53},
                {"name": "local_gateway", "host": "192.168.1.1", "port": 80}
            ],
            "components": [
                {"name": "market_data_collector", "type": "module", "path": "D/AO/AOO/A/A1/MarketDataCollector.py"},
                {"name": "technical_indicators", "type": "module", "path": "D/AO/AOO/A/A7/TechnicalIndicatorCalculator.py"}
            ],
            "alerts": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from": "",
                    "to": ""
                }
            },
            "reports": {
                "save_history": True,
                "max_history": 100,
                "export_formats": ["json", "text"]
            }
        }
    
    def _setup_logging(self) -> None:
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('health_checker.log', encoding='utf-8')
            ]
        )
    
    # ==================== 系统健康检查 ====================
    
    def check_system_health(self) -> HealthCheckResult:
        """
        执行系统健康检查
        
        Returns:
            HealthCheckResult: 系统健康检查结果
        """
        start_time = time.time()
        component_name = "system"
        
        try:
            metrics = []
            
            # CPU使用率检查
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._evaluate_metric(
                cpu_percent,
                self.config["thresholds"]["cpu_warning"],
                self.config["thresholds"]["cpu_critical"]
            )
            metrics.append(HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="%",
                status=cpu_status,
                threshold_warning=self.config["thresholds"]["cpu_warning"],
                threshold_critical=self.config["thresholds"]["cpu_critical"],
                timestamp=datetime.now(),
                description="CPU使用率"
            ))
            
            # 内存使用率检查
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_status = self._evaluate_metric(
                memory_percent,
                self.config["thresholds"]["memory_warning"],
                self.config["thresholds"]["memory_critical"]
            )
            metrics.append(HealthMetric(
                name="memory_usage",
                value=memory_percent,
                unit="%",
                status=memory_status,
                threshold_warning=self.config["thresholds"]["memory_warning"],
                threshold_critical=self.config["thresholds"]["memory_critical"],
                timestamp=datetime.now(),
                description=f"内存使用率 (可用: {memory.available / 1024**3:.2f} GB)"
            ))
            
            # 磁盘使用率检查
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            disk_status = self._evaluate_metric(
                disk_percent,
                self.config["thresholds"]["disk_warning"],
                self.config["thresholds"]["disk_critical"]
            )
            metrics.append(HealthMetric(
                name="disk_usage",
                value=disk_percent,
                unit="%",
                status=disk_status,
                threshold_warning=self.config["thresholds"]["disk_warning"],
                threshold_critical=self.config["thresholds"]["disk_critical"],
                timestamp=datetime.now(),
                description=f"磁盘使用率 (可用: {disk_usage.free / 1024**3:.2f} GB)"
            ))
            
            # 系统负载检查
            load_avg = psutil.getloadavg()[0]  # 1分钟平均负载
            cpu_count = psutil.cpu_count()
            load_percent = (load_avg / cpu_count) * 100 if cpu_count > 0 else 0
            load_status = self._evaluate_metric(
                load_percent,
                50.0,  # 负载警告阈值
                80.0   # 负载严重阈值
            )
            metrics.append(HealthMetric(
                name="system_load",
                value=load_percent,
                unit="%",
                status=load_status,
                threshold_warning=50.0,
                threshold_critical=80.0,
                timestamp=datetime.now(),
                description=f"系统负载 (1分钟平均: {load_avg:.2f})"
            ))
            
            # 确定整体状态
            overall_status = self._get_worst_status([m.status for m in metrics])
            
            duration = time.time() - start_time
            
            result = HealthCheckResult(
                check_type=CheckType.SYSTEM,
                component_name=component_name,
                status=overall_status,
                metrics=metrics,
                timestamp=datetime.now(),
                duration=duration,
                message=f"系统健康检查完成，发现 {len([m for m in metrics if m.status != HealthStatus.HEALTHY])} 个问题"
            )
            
            # 检查是否需要告警
            if overall_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self.alert_manager.send_alert(
                    f"系统健康状态异常: {overall_status.value}",
                    overall_status,
                    component_name
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"系统健康检查失败: {e}")
            return HealthCheckResult(
                check_type=CheckType.SYSTEM,
                component_name=component_name,
                status=HealthStatus.CRITICAL,
                metrics=[],
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                message=f"检查失败: {str(e)}"
            )
    
    # ==================== 服务健康检查 ====================
    
    def check_service_health(self, service_config: Dict[str, Any] = None) -> List[HealthCheckResult]:
        """
        执行服务健康检查
        
        Args:
            service_config: 服务配置，如果为None则使用默认配置
            
        Returns:
            List[HealthCheckResult]: 服务健康检查结果列表
        """
        services = service_config or self.config.get("services", [])
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_service = {
                executor.submit(self._check_single_service, service): service 
                for service in services
            }
            
            for future in as_completed(future_to_service):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    service = future_to_service[future]
                    self.logger.error(f"服务检查异常 {service['name']}: {e}")
        
        return results
    
    def _check_single_service(self, service: Dict[str, Any]) -> HealthCheckResult:
        """
        检查单个服务健康状态
        
        Args:
            service: 服务配置信息
            
        Returns:
            HealthCheckResult: 服务健康检查结果
        """
        start_time = time.time()
        component_name = service["name"]
        
        try:
            metrics = []
            
            # 端口连通性检查
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            result = sock.connect_ex((service["host"], service["port"]))
            sock.close()
            
            port_status = HealthStatus.HEALTHY if result == 0 else HealthStatus.CRITICAL
            metrics.append(HealthMetric(
                name="port_connectivity",
                value=1.0 if result == 0 else 0.0,
                unit="status",
                status=port_status,
                threshold_warning=1.0,
                threshold_critical=1.0,
                timestamp=datetime.now(),
                description=f"端口 {service['port']} 连通性"
            ))
            
            # HTTP响应检查 (如果配置了路径)
            if "path" in service and result == 0:
                response_time, http_status = self._check_http_service(service)
                http_status_enum = HealthStatus.HEALTHY if http_status == 200 else HealthStatus.WARNING
                http_response_status = self._evaluate_metric(
                    response_time,
                    self.config["thresholds"]["service_response_time_warning"],
                    self.config["thresholds"]["service_response_time_critical"]
                )
                
                metrics.append(HealthMetric(
                    name="http_response_time",
                    value=response_time,
                    unit="seconds",
                    status=http_response_status,
                    threshold_warning=self.config["thresholds"]["service_response_time_warning"],
                    threshold_critical=self.config["thresholds"]["service_response_time_critical"],
                    timestamp=datetime.now(),
                    description=f"HTTP响应时间 (状态码: {http_status})"
                ))
            
            # 确定整体状态
            overall_status = self._get_worst_status([m.status for m in metrics])
            
            duration = time.time() - start_time
            
            result = HealthCheckResult(
                check_type=CheckType.SERVICE,
                component_name=component_name,
                status=overall_status,
                metrics=metrics,
                timestamp=datetime.now(),
                duration=duration,
                message=f"服务 {service['host']}:{service['port']} 健康检查完成"
            )
            
            # 检查是否需要告警
            if overall_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self.alert_manager.send_alert(
                    f"服务 {component_name} 状态异常: {overall_status.value}",
                    overall_status,
                    component_name
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"服务 {component_name} 检查失败: {e}")
            return HealthCheckResult(
                check_type=CheckType.SERVICE,
                component_name=component_name,
                status=HealthStatus.CRITICAL,
                metrics=[],
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                message=f"检查失败: {str(e)}"
            )
    
    def _check_http_service(self, service: Dict[str, Any]) -> Tuple[float, int]:
        """
        检查HTTP服务响应
        
        Args:
            service: 服务配置信息
            
        Returns:
            Tuple[float, int]: (响应时间, HTTP状态码)
        """
        import urllib.request
        import urllib.error
        
        url = f"http://{service['host']}:{service['port']}{service.get('path', '/')}"
        
        try:
            start_time = time.time()
            with urllib.request.urlopen(url, timeout=10) as response:
                response_time = time.time() - start_time
                return response_time, response.getcode()
        except urllib.error.HTTPError as e:
            return 0.0, e.code
        except Exception as e:
            return 0.0, 500
    
    # ==================== 数据库健康检查 ====================
    
    def check_database_health(self, db_config: Dict[str, Any] = None) -> List[HealthCheckResult]:
        """
        执行数据库健康检查
        
        Args:
            db_config: 数据库配置，如果为None则使用默认配置
            
        Returns:
            List[HealthCheckResult]: 数据库健康检查结果列表
        """
        databases = db_config or self.config.get("databases", [])
        results = []
        
        for db in databases:
            try:
                result = self._check_single_database(db)
                results.append(result)
            except Exception as e:
                self.logger.error(f"数据库检查异常 {db['name']}: {e}")
        
        return results
    
    def _check_single_database(self, db_config: Dict[str, Any]) -> HealthCheckResult:
        """
        检查单个数据库健康状态
        
        Args:
            db_config: 数据库配置信息
            
        Returns:
            HealthCheckResult: 数据库健康检查结果
        """
        start_time = time.time()
        component_name = db_config["name"]
        
        try:
            metrics = []
            
            if db_config["type"] == "sqlite":
                result = self._check_sqlite_database(db_config)
            else:
                # 可以扩展支持其他数据库类型
                result = self._check_generic_database(db_config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"数据库 {component_name} 检查失败: {e}")
            return HealthCheckResult(
                check_type=CheckType.DATABASE,
                component_name=component_name,
                status=HealthStatus.CRITICAL,
                metrics=[],
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                message=f"检查失败: {str(e)}"
            )
    
    def _check_sqlite_database(self, db_config: Dict[str, Any]) -> HealthCheckResult:
        """检查SQLite数据库"""
        start_time = time.time()
        component_name = db_config["name"]
        db_path = db_config["path"]
        
        try:
            metrics = []
            
            # 数据库文件存在性检查
            file_exists = Path(db_path).exists()
            file_status = HealthStatus.HEALTHY if file_exists else HealthStatus.CRITICAL
            metrics.append(HealthMetric(
                name="file_exists",
                value=1.0 if file_exists else 0.0,
                unit="status",
                status=file_status,
                threshold_warning=1.0,
                threshold_critical=1.0,
                timestamp=datetime.now(),
                description="数据库文件存在性"
            ))
            
            if file_exists:
                # 数据库连接检查
                connect_start = time.time()
                conn = sqlite3.connect(db_path, timeout=5.0)
                connect_time = time.time() - connect_start
                
                connect_status = self._evaluate_metric(
                    connect_time,
                    self.config["thresholds"]["db_connection_timeout_warning"],
                    self.config["thresholds"]["db_connection_timeout_critical"]
                )
                
                metrics.append(HealthMetric(
                    name="connection_time",
                    value=connect_time,
                    unit="seconds",
                    status=connect_status,
                    threshold_warning=self.config["thresholds"]["db_connection_timeout_warning"],
                    threshold_critical=self.config["thresholds"]["db_connection_timeout_critical"],
                    timestamp=datetime.now(),
                    description="数据库连接时间"
                ))
                
                # 数据库大小检查
                db_size = Path(db_path).stat().st_size / (1024 * 1024)  # MB
                size_status = HealthStatus.HEALTHY
                if db_size > 1000:  # 1GB
                    size_status = HealthStatus.WARNING
                if db_size > 5000:  # 5GB
                    size_status = HealthStatus.CRITICAL
                
                metrics.append(HealthMetric(
                    name="database_size",
                    value=db_size,
                    unit="MB",
                    status=size_status,
                    threshold_warning=1000.0,
                    threshold_critical=5000.0,
                    timestamp=datetime.now(),
                    description="数据库文件大小"
                ))
                
                # 简单查询测试
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    query_status = HealthStatus.HEALTHY
                    query_message = "查询测试成功"
                except Exception as e:
                    query_status = HealthStatus.CRITICAL
                    query_message = f"查询测试失败: {str(e)}"
                
                metrics.append(HealthMetric(
                    name="query_test",
                    value=1.0 if query_status == HealthStatus.HEALTHY else 0.0,
                    unit="status",
                    status=query_status,
                    threshold_warning=1.0,
                    threshold_critical=1.0,
                    timestamp=datetime.now(),
                    description=query_message
                ))
                
                conn.close()
            
            # 确定整体状态
            overall_status = self._get_worst_status([m.status for m in metrics])
            
            duration = time.time() - start_time
            
            result = HealthCheckResult(
                check_type=CheckType.DATABASE,
                component_name=component_name,
                status=overall_status,
                metrics=metrics,
                timestamp=datetime.now(),
                duration=duration,
                message=f"数据库 {db_path} 健康检查完成"
            )
            
            # 检查是否需要告警
            if overall_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self.alert_manager.send_alert(
                    f"数据库 {component_name} 状态异常: {overall_status.value}",
                    overall_status,
                    component_name
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"SQLite数据库检查失败: {e}")
            return HealthCheckResult(
                check_type=CheckType.DATABASE,
                component_name=component_name,
                status=HealthStatus.CRITICAL,
                metrics=[],
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                message=f"检查失败: {str(e)}"
            )
    
    def _check_generic_database(self, db_config: Dict[str, Any]) -> HealthCheckResult:
        """检查通用数据库 (可扩展)"""
        # 这里可以添加对其他数据库类型的支持
        # 如 MySQL, PostgreSQL, MongoDB 等
        return HealthCheckResult(
            check_type=CheckType.DATABASE,
            component_name=db_config["name"],
            status=HealthStatus.UNKNOWN,
            metrics=[],
            timestamp=datetime.now(),
            duration=0.0,
            message=f"暂不支持的数据库类型: {db_config['type']}"
        )
    
    # ==================== 网络健康检查 ====================
    
    def check_network_health(self, network_config: Dict[str, Any] = None) -> List[HealthCheckResult]:
        """
        执行网络健康检查
        
        Args:
            network_config: 网络配置，如果为None则使用默认配置
            
        Returns:
            List[HealthCheckResult]: 网络健康检查结果列表
        """
        targets = network_config or self.config.get("network_targets", [])
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_target = {
                executor.submit(self._check_single_network, target): target 
                for target in targets
            }
            
            for future in as_completed(future_to_target):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    target = future_to_target[future]
                    self.logger.error(f"网络检查异常 {target['name']}: {e}")
        
        return results
    
    def _check_single_network(self, target: Dict[str, Any]) -> HealthCheckResult:
        """
        检查单个网络目标
        
        Args:
            target: 网络目标配置
            
        Returns:
            HealthCheckResult: 网络健康检查结果
        """
        start_time = time.time()
        component_name = target["name"]
        
        try:
            metrics = []
            
            # Ping延迟检查
            ping_time = self._ping_host(target["host"])
            ping_status = self._evaluate_metric(
                ping_time,
                self.config["thresholds"]["network_latency_warning"],
                self.config["thresholds"]["network_latency_critical"]
            )
            
            metrics.append(HealthMetric(
                name="ping_latency",
                value=ping_time,
                unit="milliseconds",
                status=ping_status,
                threshold_warning=self.config["thresholds"]["network_latency_warning"],
                threshold_critical=self.config["thresholds"]["network_latency_critical"],
                timestamp=datetime.now(),
                description=f"到 {target['host']} 的ping延迟"
            ))
            
            # 端口连通性检查
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            result = sock.connect_ex((target["host"], target["port"]))
            sock.close()
            
            port_status = HealthStatus.HEALTHY if result == 0 else HealthStatus.CRITICAL
            metrics.append(HealthMetric(
                name="port_connectivity",
                value=1.0 if result == 0 else 0.0,
                unit="status",
                status=port_status,
                threshold_warning=1.0,
                threshold_critical=1.0,
                timestamp=datetime.now(),
                description=f"端口 {target['port']} 连通性"
            ))
            
            # 确定整体状态
            overall_status = self._get_worst_status([m.status for m in metrics])
            
            duration = time.time() - start_time
            
            result = HealthCheckResult(
                check_type=CheckType.NETWORK,
                component_name=component_name,
                status=overall_status,
                metrics=metrics,
                timestamp=datetime.now(),
                duration=duration,
                message=f"网络目标 {target['host']}:{target['port']} 健康检查完成"
            )
            
            # 检查是否需要告警
            if overall_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self.alert_manager.send_alert(
                    f"网络目标 {component_name} 状态异常: {overall_status.value}",
                    overall_status,
                    component_name
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"网络目标 {component_name} 检查失败: {e}")
            return HealthCheckResult(
                check_type=CheckType.NETWORK,
                component_name=component_name,
                status=HealthStatus.CRITICAL,
                metrics=[],
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                message=f"检查失败: {str(e)}"
            )
    
    def _ping_host(self, host: str) -> float:
        """
        Ping主机获取延迟
        
        Args:
            host: 目标主机地址
            
        Returns:
            float: 延迟时间(毫秒)
        """
        try:
            # 使用系统ping命令
            if subprocess.platform.system().lower() == "windows":
                cmd = ["ping", "-n", "1", host]
            else:
                cmd = ["ping", "-c", "1", host]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # 解析ping结果中的时间
                output = result.stdout
                if "time=" in output:
                    time_str = output.split("time=")[1].split()[0]
                    return float(time_str)
                elif "时间=" in output:  # Windows中文版
                    time_str = output.split("时间=")[1].split()[0]
                    return float(time_str)
            
            return 999.0  # 超时或失败
            
        except Exception:
            return 999.0  # 异常情况返回大值
    
    # ==================== 组件健康检查 ====================
    
    def check_component_health(self, component_config: Dict[str, Any] = None) -> List[HealthCheckResult]:
        """
        执行组件健康检查
        
        Args:
            component_config: 组件配置，如果为None则使用默认配置
            
        Returns:
            List[HealthCheckResult]: 组件健康检查结果列表
        """
        components = component_config or self.config.get("components", [])
        results = []
        
        for component in components:
            try:
                result = self._check_single_component(component)
                results.append(result)
            except Exception as e:
                self.logger.error(f"组件检查异常 {component['name']}: {e}")
        
        return results
    
    def _check_single_component(self, component: Dict[str, Any]) -> HealthCheckResult:
        """
        检查单个组件健康状态
        
        Args:
            component: 组件配置信息
            
        Returns:
            HealthCheckResult: 组件健康检查结果
        """
        start_time = time.time()
        component_name = component["name"]
        
        try:
            metrics = []
            
            if component["type"] == "module":
                result = self._check_python_module(component)
            elif component["type"] == "process":
                result = self._check_process(component)
            else:
                result = HealthCheckResult(
                    check_type=CheckType.COMPONENT,
                    component_name=component_name,
                    status=HealthStatus.UNKNOWN,
                    metrics=[],
                    timestamp=datetime.now(),
                    duration=0.0,
                    message=f"未知的组件类型: {component['type']}"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"组件 {component_name} 检查失败: {e}")
            return HealthCheckResult(
                check_type=CheckType.COMPONENT,
                component_name=component_name,
                status=HealthStatus.CRITICAL,
                metrics=[],
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                message=f"检查失败: {str(e)}"
            )
    
    def _check_python_module(self, component: Dict[str, Any]) -> HealthCheckResult:
        """检查Python模块组件"""
        start_time = time.time()
        component_name = component["name"]
        module_path = component["path"]
        
        try:
            metrics = []
            
            # 文件存在性检查
            file_exists = Path(module_path).exists()
            file_status = HealthStatus.HEALTHY if file_exists else HealthStatus.CRITICAL
            metrics.append(HealthMetric(
                name="file_exists",
                value=1.0 if file_exists else 0.0,
                unit="status",
                status=file_status,
                threshold_warning=1.0,
                threshold_critical=1.0,
                timestamp=datetime.now(),
                description="模块文件存在性"
            ))
            
            if file_exists:
                # 文件大小检查
                file_size = Path(module_path).stat().st_size
                size_status = HealthStatus.HEALTHY
                if file_size > 100000:  # 100KB
                    size_status = HealthStatus.WARNING
                if file_size > 1000000:  # 1MB
                    size_status = HealthStatus.CRITICAL
                
                metrics.append(HealthMetric(
                    name="file_size",
                    value=file_size,
                    unit="bytes",
                    status=size_status,
                    threshold_warning=100000.0,
                    threshold_critical=1000000.0,
                    timestamp=datetime.now(),
                    description="模块文件大小"
                ))
                
                # 语法检查
                try:
                    with open(module_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    compile(code, module_path, 'exec')
                    syntax_status = HealthStatus.HEALTHY
                    syntax_message = "语法检查通过"
                except SyntaxError as e:
                    syntax_status = HealthStatus.CRITICAL
                    syntax_message = f"语法错误: {str(e)}"
                except Exception as e:
                    syntax_status = HealthStatus.WARNING
                    syntax_message = f"语法检查异常: {str(e)}"
                
                metrics.append(HealthMetric(
                    name="syntax_check",
                    value=1.0 if syntax_status == HealthStatus.HEALTHY else 0.0,
                    unit="status",
                    status=syntax_status,
                    threshold_warning=1.0,
                    threshold_critical=1.0,
                    timestamp=datetime.now(),
                    description=syntax_message
                ))
            
            # 确定整体状态
            overall_status = self._get_worst_status([m.status for m in metrics])
            
            duration = time.time() - start_time
            
            result = HealthCheckResult(
                check_type=CheckType.COMPONENT,
                component_name=component_name,
                status=overall_status,
                metrics=metrics,
                timestamp=datetime.now(),
                duration=duration,
                message=f"组件 {module_path} 健康检查完成"
            )
            
            # 检查是否需要告警
            if overall_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self.alert_manager.send_alert(
                    f"组件 {component_name} 状态异常: {overall_status.value}",
                    overall_status,
                    component_name
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Python模块检查失败: {e}")
            return HealthCheckResult(
                check_type=CheckType.COMPONENT,
                component_name=component_name,
                status=HealthStatus.CRITICAL,
                metrics=[],
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                message=f"检查失败: {str(e)}"
            )
    
    def _check_process(self, component: Dict[str, Any]) -> HealthCheckResult:
        """检查进程组件"""
        # 这里可以添加进程检查逻辑
        # 如检查特定进程是否运行、CPU使用率等
        return HealthCheckResult(
            check_type=CheckType.COMPONENT,
            component_name=component["name"],
            status=HealthStatus.UNKNOWN,
            metrics=[],
            timestamp=datetime.now(),
            duration=0.0,
            message="进程检查功能待实现"
        )
    
    # ==================== 健康状态评估 ====================
    
    def evaluate_overall_health(self, results: List[HealthCheckResult]) -> HealthStatus:
        """
        评估整体健康状态
        
        Args:
            results: 健康检查结果列表
            
        Returns:
            HealthStatus: 整体健康状态
        """
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results]
        return self._get_worst_status(statuses)
    
    def _get_worst_status(self, statuses: List[HealthStatus]) -> HealthStatus:
        """
        获取最严重的健康状态
        
        Args:
            statuses: 状态列表
            
        Returns:
            HealthStatus: 最严重的状态
        """
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.HEALTHY in statuses:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _evaluate_metric(self, value: float, warning_threshold: float, 
                        critical_threshold: float) -> HealthStatus:
        """
        评估指标健康状态
        
        Args:
            value: 指标值
            warning_threshold: 警告阈值
            critical_threshold: 严重阈值
            
        Returns:
            HealthStatus: 健康状态
        """
        if value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    # ==================== 健康报告生成 ====================
    
    def generate_health_report(self, results: List[HealthCheckResult]) -> HealthReport:
        """
        生成健康报告
        
        Args:
            results: 健康检查结果列表
            
        Returns:
            HealthReport: 健康报告
        """
        overall_status = self.evaluate_overall_health(results)
        
        # 统计各类状态的数量
        healthy_count = len([r for r in results if r.status == HealthStatus.HEALTHY])
        warning_count = len([r for r in results if r.status == HealthStatus.WARNING])
        critical_count = len([r for r in results if r.status == HealthStatus.CRITICAL])
        
        # 生成建议
        recommendations = self._generate_recommendations(results)
        
        report = HealthReport(
            overall_status=overall_status,
            timestamp=datetime.now(),
            checks_performed=len(results),
            healthy_components=healthy_count,
            warning_components=warning_count,
            critical_components=critical_count,
            results=results,
            recommendations=recommendations
        )
        
        # 保存报告历史
        if self.config["reports"]["save_history"]:
            self._save_report_history(report)
        
        return report
    
    def _generate_recommendations(self, results: List[HealthCheckResult]) -> List[str]:
        """
        根据检查结果生成建议
        
        Args:
            results: 健康检查结果列表
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        for result in results:
            if result.status == HealthStatus.CRITICAL:
                recommendations.append(f"紧急: {result.component_name} 存在严重问题，需要立即处理")
            elif result.status == HealthStatus.WARNING:
                recommendations.append(f"注意: {result.component_name} 状态不佳，建议关注")
            
            # 针对具体指标给出建议
            for metric in result.metrics:
                if metric.status == HealthStatus.CRITICAL:
                    if metric.name == "cpu_usage":
                        recommendations.append("CPU使用率过高，建议优化应用程序或增加硬件资源")
                    elif metric.name == "memory_usage":
                        recommendations.append("内存使用率过高，建议释放内存或增加内存容量")
                    elif metric.name == "disk_usage":
                        recommendations.append("磁盘空间不足，建议清理文件或增加存储容量")
                    elif metric.name == "ping_latency":
                        recommendations.append("网络延迟过高，建议检查网络连接或联系网络管理员")
        
        return recommendations
    
    def _save_report_history(self, report: HealthReport) -> None:
        """保存报告历史"""
        try:
            self.check_history.append(report)
            
            # 限制历史记录数量
            max_history = self.config["reports"]["max_history"]
            if len(self.check_history) > max_history:
                self.check_history = self.check_history[-max_history:]
                
        except Exception as e:
            self.logger.error(f"保存报告历史失败: {e}")
    
    def export_report(self, report: HealthReport, format_type: str = "json", 
                     file_path: str = None) -> str:
        """
        导出健康报告
        
        Args:
            report: 健康报告
            format_type: 导出格式 ("json" 或 "text")
            file_path: 文件路径，如果为None则自动生成
            
        Returns:
            str: 导出的文件路径
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"health_report_{timestamp}.{format_type}"
        
        try:
            if format_type.lower() == "json":
                self._export_json_report(report, file_path)
            elif format_type.lower() == "text":
                self._export_text_report(report, file_path)
            else:
                raise ValueError(f"不支持的导出格式: {format_type}")
            
            self.logger.info(f"健康报告已导出到: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"导出报告失败: {e}")
            raise
    
    def _export_json_report(self, report: HealthReport, file_path: str) -> None:
        """导出JSON格式报告"""
        def serialize_object(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        report_dict = asdict(report)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2, default=serialize_object)
    
    def _export_text_report(self, report: HealthReport, file_path: str) -> None:
        """导出文本格式报告"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("系统健康检查报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"报告生成时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"整体健康状态: {report.overall_status.value.upper()}\n")
            f.write(f"检查组件数量: {report.checks_performed}\n")
            f.write(f"  - 健康: {report.healthy_components}\n")
            f.write(f"  - 警告: {report.warning_components}\n")
            f.write(f"  - 严重: {report.critical_components}\n\n")
            
            f.write("详细检查结果:\n")
            f.write("-" * 80 + "\n")
            
            for result in report.results:
                f.write(f"\n[{result.check_type.value.upper()}] {result.component_name}\n")
                f.write(f"状态: {result.status.value.upper()}\n")
                f.write(f"耗时: {result.duration:.2f}秒\n")
                f.write(f"消息: {result.message}\n")
                
                if result.metrics:
                    f.write("指标详情:\n")
                    for metric in result.metrics:
                        f.write(f"  - {metric.name}: {metric.value} {metric.unit} "
                               f"({metric.status.value})\n")
                        if metric.description:
                            f.write(f"    {metric.description}\n")
                f.write("-" * 40 + "\n")
            
            if report.recommendations:
                f.write("\n建议:\n")
                f.write("-" * 40 + "\n")
                for i, rec in enumerate(report.recommendations, 1):
                    f.write(f"{i}. {rec}\n")
    
    # ==================== 健康检查调度 ====================
    
    def start_scheduled_checks(self) -> None:
        """启动定时健康检查"""
        self.logger.info("启动定时健康检查...")
        
        # 为每种检查类型创建定时任务
        for check_type, interval in self.config["check_intervals"].items():
            thread = threading.Thread(
                target=self._scheduled_check_worker,
                args=(check_type, interval),
                daemon=True
            )
            thread.start()
            self.running_checks[check_type] = thread
    
    def stop_scheduled_checks(self) -> None:
        """停止定时健康检查"""
        self.logger.info("停止定时健康检查...")
        # 由于使用daemon线程，主线程结束时会自动停止
        self.running_checks.clear()
    
    def _scheduled_check_worker(self, check_type: str, interval: int) -> None:
        """
        定时检查工作线程
        
        Args:
            check_type: 检查类型
            interval: 检查间隔(秒)
        """
        while True:
            try:
                if check_type == "system":
                    result = self.check_system_health()
                elif check_type == "service":
                    results = self.check_service_health()
                    # 这里可以处理服务检查结果
                elif check_type == "database":
                    results = self.check_database_health()
                    # 这里可以处理数据库检查结果
                elif check_type == "network":
                    results = self.check_network_health()
                    # 这里可以处理网络检查结果
                elif check_type == "component":
                    results = self.check_component_health()
                    # 这里可以处理组件检查结果
                
                self.logger.debug(f"{check_type} 检查完成")
                
            except Exception as e:
                self.logger.error(f"{check_type} 检查异常: {e}")
            
            time.sleep(interval)
    
    # ==================== 完整健康检查 ====================
    
    def perform_comprehensive_health_check(self) -> HealthReport:
        """
        执行全面的健康检查
        
        Returns:
            HealthReport: 综合健康报告
        """
        self.logger.info("开始执行全面健康检查...")
        
        all_results = []
        
        # 系统健康检查
        try:
            system_result = self.check_system_health()
            all_results.append(system_result)
            self.logger.info("系统健康检查完成")
        except Exception as e:
            self.logger.error(f"系统健康检查失败: {e}")
        
        # 服务健康检查
        try:
            service_results = self.check_service_health()
            all_results.extend(service_results)
            self.logger.info(f"服务健康检查完成，检查了 {len(service_results)} 个服务")
        except Exception as e:
            self.logger.error(f"服务健康检查失败: {e}")
        
        # 数据库健康检查
        try:
            db_results = self.check_database_health()
            all_results.extend(db_results)
            self.logger.info(f"数据库健康检查完成，检查了 {len(db_results)} 个数据库")
        except Exception as e:
            self.logger.error(f"数据库健康检查失败: {e}")
        
        # 网络健康检查
        try:
            network_results = self.check_network_health()
            all_results.extend(network_results)
            self.logger.info(f"网络健康检查完成，检查了 {len(network_results)} 个网络目标")
        except Exception as e:
            self.logger.error(f"网络健康检查失败: {e}")
        
        # 组件健康检查
        try:
            component_results = self.check_component_health()
            all_results.extend(component_results)
            self.logger.info(f"组件健康检查完成，检查了 {len(component_results)} 个组件")
        except Exception as e:
            self.logger.error(f"组件健康检查失败: {e}")
        
        # 生成综合报告
        report = self.generate_health_report(all_results)
        
        self.logger.info(f"全面健康检查完成，整体状态: {report.overall_status.value}")
        
        return report
    
    # ==================== 工具方法 ====================
    
    def get_check_history(self, limit: int = 10) -> List[HealthReport]:
        """
        获取检查历史
        
        Args:
            limit: 返回记录数量限制
            
        Returns:
            List[HealthReport]: 历史报告列表
        """
        return self.check_history[-limit:] if self.check_history else []
    
    def clear_check_history(self) -> None:
        """清空检查历史"""
        self.check_history.clear()
        self.logger.info("检查历史已清空")


# ==================== 测试用例 ====================

def test_health_checker():
    """健康检查器测试函数"""
    print("开始测试健康检查器...")
    
    # 创建健康检查器实例
    checker = HealthChecker()
    
    # 测试系统健康检查
    print("\n1. 测试系统健康检查:")
    system_result = checker.check_system_health()
    print(f"状态: {system_result.status.value}")
    print(f"耗时: {system_result.duration:.2f}秒")
    for metric in system_result.metrics:
        print(f"  - {metric.name}: {metric.value} {metric.unit} ({metric.status.value})")
    
    # 测试服务健康检查
    print("\n2. 测试服务健康检查:")
    service_results = checker.check_service_health()
    for result in service_results:
        print(f"服务 {result.component_name}: {result.status.value}")
        for metric in result.metrics:
            print(f"  - {metric.name}: {metric.value} {metric.unit} ({metric.status.value})")
    
    # 测试数据库健康检查
    print("\n3. 测试数据库健康检查:")
    db_results = checker.check_database_health()
    for result in db_results:
        print(f"数据库 {result.component_name}: {result.status.value}")
        for metric in result.metrics:
            print(f"  - {metric.name}: {metric.value} {metric.unit} ({metric.status.value})")
    
    # 测试网络健康检查
    print("\n4. 测试网络健康检查:")
    network_results = checker.check_network_health()
    for result in network_results:
        print(f"网络目标 {result.component_name}: {result.status.value}")
        for metric in result.metrics:
            print(f"  - {metric.name}: {metric.value} {metric.unit} ({metric.status.value})")
    
    # 测试组件健康检查
    print("\n5. 测试组件健康检查:")
    component_results = checker.check_component_health()
    for result in component_results:
        print(f"组件 {result.component_name}: {result.status.value}")
        for metric in result.metrics:
            print(f"  - {metric.name}: {metric.value} {metric.unit} ({metric.status.value})")
    
    # 测试综合健康检查
    print("\n6. 测试综合健康检查:")
    comprehensive_report = checker.perform_comprehensive_health_check()
    print(f"整体状态: {comprehensive_report.overall_status.value}")
    print(f"检查数量: {comprehensive_report.checks_performed}")
    print(f"健康组件: {comprehensive_report.healthy_components}")
    print(f"警告组件: {comprehensive_report.warning_components}")
    print(f"严重组件: {comprehensive_report.critical_components}")
    
    # 测试报告导出
    print("\n7. 测试报告导出:")
    try:
        json_file = checker.export_report(comprehensive_report, "json", "test_health_report.json")
        text_file = checker.export_report(comprehensive_report, "text", "test_health_report.txt")
        print(f"JSON报告已导出: {json_file}")
        print(f"文本报告已导出: {text_file}")
    except Exception as e:
        print(f"报告导出失败: {e}")
    
    print("\n健康检查器测试完成!")


def demo_health_checker():
    """健康检查器演示函数"""
    print("健康检查器演示")
    print("=" * 50)
    
    # 创建自定义配置
    config = {
        "thresholds": {
            "cpu_warning": 60.0,
            "cpu_critical": 85.0,
            "memory_warning": 70.0,
            "memory_critical": 90.0,
            "disk_warning": 75.0,
            "disk_critical": 85.0,
        },
        "services": [
            {"name": "web_service", "host": "localhost", "port": 80, "path": "/"},
            {"name": "api_service", "host": "localhost", "port": 8080, "path": "/health"},
        ],
        "databases": [
            {"name": "main_database", "type": "sqlite", "path": "demo.db"},
        ],
        "network_targets": [
            {"name": "dns_server", "host": "8.8.8.8", "port": 53},
        ],
        "components": [
            {"name": "demo_module", "type": "module", "path": __file__},
        ]
    }
    
    # 创建健康检查器
    checker = HealthChecker(config)
    
    # 执行演示检查
    print("执行演示检查...")
    report = checker.perform_comprehensive_health_check()
    
    # 显示结果
    print(f"\n检查结果:")
    print(f"整体状态: {report.overall_status.value}")
    print(f"检查组件: {report.checks_performed}")
    print(f"健康: {report.healthy_components}, 警告: {report.warning_components}, 严重: {report.critical_components}")
    
    if report.recommendations:
        print(f"\n建议:")
        for rec in report.recommendations:
            print(f"- {rec}")
    
    # 导出报告
    try:
        report_file = checker.export_report(report, "json", "demo_health_report.json")
        print(f"\n报告已保存到: {report_file}")
    except Exception as e:
        print(f"报告导出失败: {e}")


if __name__ == "__main__":
    # 运行测试
    test_health_checker()
    
    print("\n" + "=" * 80 + "\n")
    
    # 运行演示
    demo_health_checker()