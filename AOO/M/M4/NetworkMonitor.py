#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络监控器 (NetworkMonitor)

实现全面的网络监控功能，包括连接监控、延迟监控、带宽监控、错误监控、
安全监控、拓扑发现、流量分析、性能优化和监控报告。


创建时间: 2025-11-05
版本: 1.0.0
"""

import asyncio
import json
import logging
import psutil
import socket
import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from ipaddress import ip_network, ip_address
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import statistics
import hashlib
import ssl
import requests
from urllib.parse import urlparse
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NetworkConnection:
    """网络连接数据类"""
    source_ip: str
    source_port: int
    dest_ip: str
    dest_port: int
    protocol: str
    status: str
    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    latency: float = 0.0


@dataclass
class NetworkMetrics:
    """网络指标数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    network_io: Dict[str, int]
    connections_count: int
    bandwidth_usage: float
    latency: float
    packet_loss: float
    errors_count: int


@dataclass
class SecurityAlert:
    """安全警报数据类"""
    alert_id: str
    timestamp: datetime
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    alert_type: str
    description: str
    source_ip: Optional[str] = None
    target_ip: Optional[str] = None
    mitigation: Optional[str] = None


@dataclass
class NetworkTopology:
    """网络拓扑数据类"""
    node_id: str
    ip_address: str
    hostname: str
    mac_address: Optional[str] = None
    device_type: str = "unknown"
    status: str = "unknown"
    neighbors: List[str] = field(default_factory=list)
    bandwidth: Optional[float] = None


@dataclass
class TrafficAnalysis:
    """流量分析数据类"""
    protocol: str
    source_ips: Dict[str, int]
    dest_ips: Dict[str, int]
    total_bytes: int
    packet_count: int
    duration: float
    avg_packet_size: float
    peak_bandwidth: float
    flow_pattern: str


class NetworkMonitor:
    """
    网络监控器主类
    
    提供全面的网络监控功能，包括连接监控、延迟监控、带宽监控、
    错误监控、安全监控、拓扑发现、流量分析、性能优化和监控报告。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化网络监控器
        
        Args:
            config: 配置字典，包含监控参数
        """
        self.config = config or self._default_config()
        self.is_monitoring = False
        self.monitor_thread = None
        self.data_lock = threading.RLock()
        
        # 数据存储
        self.connection_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metrics_history: deque = deque(maxlen=10000)
        self.security_alerts: List[SecurityAlert] = []
        self.topology_map: Dict[str, NetworkTopology] = {}
        self.traffic_analysis: Dict[str, TrafficAnalysis] = {}
        
        # 监控回调函数
        self.alert_callbacks: List[Callable[[SecurityAlert], None]] = []
        self.metrics_callbacks: List[Callable[[NetworkMetrics], None]] = []
        
        # 性能优化
        self.optimization_rules: List[Dict[str, Any]] = []
        self.performance_baseline: Optional[Dict[str, float]] = None
        
        logger.info("网络监控器初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "monitor_interval": 1.0,  # 监控间隔（秒）
            "connection_timeout": 5.0,  # 连接超时时间（秒）
            "latency_threshold": 100.0,  # 延迟阈值（毫秒）
            "bandwidth_threshold": 80.0,  # 带宽使用率阈值（%）
            "packet_loss_threshold": 5.0,  # 丢包率阈值（%）
            "error_threshold": 10,  # 错误计数阈值
            "security_scan_interval": 300,  # 安全扫描间隔（秒）
            "topology_discovery_interval": 600,  # 拓扑发现间隔（秒）
            "traffic_analysis_interval": 60,  # 流量分析间隔（秒）
            "max_connections": 1000,  # 最大连接数
            "log_level": "INFO",  # 日志级别
            "enable_security_monitoring": True,  # 启用安全监控
            "enable_topology_discovery": True,  # 启用拓扑发现
            "enable_traffic_analysis": True,  # 启用流量分析
            "enable_performance_optimization": True,  # 启用性能优化
            "monitored_hosts": [],  # 监控的主机列表
            "blocked_ips": [],  # 阻止的IP列表
            "trusted_ips": [],  # 信任的IP列表
        }
    
    # ==================== 网络连接监控 ====================
    
    def get_active_connections(self) -> List[NetworkConnection]:
        """
        获取当前活跃的网络连接
        
        Returns:
            List[NetworkConnection]: 活跃连接列表
        """
        connections = []
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == 'ESTABLISHED':
                    connection = NetworkConnection(
                        source_ip=conn.laddr.ip if conn.laddr else "0.0.0.0",
                        source_port=conn.laddr.port if conn.laddr else 0,
                        dest_ip=conn.raddr.ip if conn.raddr else "0.0.0.0",
                        dest_port=conn.raddr.port if conn.raddr else 0,
                        protocol=conn.type.name.lower(),
                        status=conn.status.lower()
                    )
                    connections.append(connection)
        except Exception as e:
            logger.error(f"获取活跃连接失败: {e}")
        
        return connections
    
    def monitor_connection_health(self) -> Dict[str, Any]:
        """
        监控连接健康状态
        
        Returns:
            Dict[str, Any]: 连接健康状态信息
        """
        health_info = {
            "total_connections": 0,
            "established_connections": 0,
            "listening_ports": 0,
            "connection_rate": 0.0,
            "failed_connections": 0,
            "timestamp": datetime.now()
        }
        
        try:
            connections = psutil.net_connections(kind='inet')
            health_info["total_connections"] = len(connections)
            
            for conn in connections:
                if conn.status == 'ESTABLISHED':
                    health_info["established_connections"] += 1
                elif conn.status == 'LISTEN':
                    health_info["listening_ports"] += 1
            
            # 计算连接率
            if health_info["total_connections"] > 0:
                health_info["connection_rate"] = (
                    health_info["established_connections"] / health_info["total_connections"]
                ) * 100
            
        except Exception as e:
            logger.error(f"连接健康监控失败: {e}")
            health_info["failed_connections"] = 1
        
        return health_info
    
    # ==================== 网络延迟监控 ====================
    
    def ping_host(self, host: str, timeout: float = 3.0) -> Tuple[float, bool]:
        """
        Ping指定主机
        
        Args:
            host: 目标主机地址
            timeout: 超时时间（秒）
            
        Returns:
            Tuple[float, bool]: (延迟时间, 是否成功)
        """
        try:
            # 使用系统ping命令
            if socket.gethostbyname(host) == host:
                # IPv4地址
                result = subprocess.run(
                    ['ping', '-c', '1', '-W', str(int(timeout * 1000)), host],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            else:
                # 主机名或IPv6地址
                result = subprocess.run(
                    ['ping6', '-c', '1', '-W', str(int(timeout * 1000)), host],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            
            if result.returncode == 0:
                # 解析ping结果获取延迟时间
                output = result.stdout
                if 'time=' in output:
                    time_part = output.split('time=')[1].split()[0]
                    latency = float(time_part)
                    return latency, True
            
            return 0.0, False
            
        except Exception as e:
            logger.error(f"Ping {host} 失败: {e}")
            return 0.0, False
    
    def monitor_network_latency(self, targets: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        监控网络延迟
        
        Args:
            targets: 目标主机列表
            
        Returns:
            Dict[str, Dict[str, float]]: 延迟监控结果
        """
        if targets is None:
            targets = self.config.get("monitored_hosts", ["8.8.8.8", "114.114.114.114"])
        
        latency_results = {}
        
        for target in targets:
            try:
                # 连续ping 5次计算平均延迟
                latencies = []
                success_count = 0
                
                for _ in range(5):
                    latency, success = self.ping_host(target)
                    if success:
                        latencies.append(latency)
                        success_count += 1
                    time.sleep(0.1)
                
                if latencies:
                    latency_results[target] = {
                        "avg_latency": statistics.mean(latencies),
                        "min_latency": min(latencies),
                        "max_latency": max(latencies),
                        "success_rate": (success_count / 5) * 100,
                        "packet_loss": ((5 - success_count) / 5) * 100
                    }
                else:
                    latency_results[target] = {
                        "avg_latency": 0.0,
                        "min_latency": 0.0,
                        "max_latency": 0.0,
                        "success_rate": 0.0,
                        "packet_loss": 100.0
                    }
                    
            except Exception as e:
                logger.error(f"监控 {target} 延迟失败: {e}")
                latency_results[target] = {
                    "avg_latency": 0.0,
                    "min_latency": 0.0,
                    "max_latency": 0.0,
                    "success_rate": 0.0,
                    "packet_loss": 100.0
                }
        
        return latency_results
    
    # ==================== 网络带宽使用监控 ====================
    
    def get_network_interface_stats(self) -> Dict[str, Dict[str, int]]:
        """
        获取网络接口统计信息
        
        Returns:
            Dict[str, Dict[str, int]]: 网络接口统计信息
        """
        stats = {}
        try:
            for interface, stats_data in psutil.net_io_counters(pernic=True).items():
                stats[interface] = {
                    "bytes_sent": stats_data.bytes_sent,
                    "bytes_recv": stats_data.bytes_recv,
                    "packets_sent": stats_data.packets_sent,
                    "packets_recv": stats_data.packets_recv,
                    "errin": stats_data.errin,
                    "errout": stats_data.errout,
                    "dropin": stats_data.dropin,
                    "dropout": stats_data.dropout
                }
        except Exception as e:
            logger.error(f"获取网络接口统计失败: {e}")
        
        return stats
    
    def monitor_bandwidth_usage(self, interval: float = 1.0) -> Dict[str, Any]:
        """
        监控带宽使用情况
        
        Args:
            interval: 监控间隔（秒）
            
        Returns:
            Dict[str, Any]: 带宽使用信息
        """
        try:
            # 获取初始统计
            initial_stats = self.get_network_interface_stats()
            time.sleep(interval)
            
            # 获取最终统计
            final_stats = self.get_network_interface_stats()
            
            bandwidth_info = {
                "interfaces": {},
                "total_bandwidth_usage": 0.0,
                "timestamp": datetime.now()
            }
            
            for interface in initial_stats:
                if interface in final_stats:
                    init = initial_stats[interface]
                    final = final_stats[interface]
                    
                    bytes_sent_rate = (final["bytes_sent"] - init["bytes_sent"]) / interval
                    bytes_recv_rate = (final["bytes_recv"] - init["bytes_recv"]) / interval
                    
                    # 转换为Mbps
                    send_mbps = (bytes_sent_rate * 8) / (1024 * 1024)
                    recv_mbps = (bytes_recv_rate * 8) / (1024 * 1024)
                    
                    bandwidth_info["interfaces"][interface] = {
                        "send_rate_mbps": send_mbps,
                        "recv_rate_mbps": recv_mbps,
                        "total_rate_mbps": send_mbps + recv_mbps,
                        "bytes_sent": final["bytes_sent"],
                        "bytes_recv": final["bytes_recv"],
                        "errors": final["errin"] + final["errout"],
                        "packet_loss": final["dropin"] + final["dropout"]
                    }
                    
                    bandwidth_info["total_bandwidth_usage"] += send_mbps + recv_mbps
            
            return bandwidth_info
            
        except Exception as e:
            logger.error(f"带宽监控失败: {e}")
            return {"error": str(e), "timestamp": datetime.now()}
    
    # ==================== 网络错误监控 ====================
    
    def monitor_network_errors(self) -> Dict[str, Any]:
        """
        监控网络错误
        
        Returns:
            Dict[str, Any]: 网络错误信息
        """
        error_info = {
            "interface_errors": {},
            "connection_errors": 0,
            "dns_errors": 0,
            "timeout_errors": 0,
            "total_errors": 0,
            "timestamp": datetime.now()
        }
        
        try:
            # 获取接口错误统计
            stats = self.get_network_interface_stats()
            for interface, data in stats.items():
                error_count = data["errin"] + data["errout"] + data["dropin"] + data["dropout"]
                error_info["interface_errors"][interface] = error_count
                error_info["total_errors"] += error_count
            
            # 检查连接错误
            connections = psutil.net_connections(kind='inet')
            for conn in connections:
                if conn.status in ['TIME_WAIT', 'CLOSE_WAIT', 'LAST_ACK']:
                    error_info["connection_errors"] += 1
            
            # 检查DNS错误（简单实现）
            try:
                socket.gethostbyname("nonexistent.invalid")
            except socket.gaierror:
                error_info["dns_errors"] += 1
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"网络错误监控失败: {e}")
            error_info["error"] = str(e)
        
        return error_info
    
    # ==================== 网络安全监控 ====================
    
    def scan_port(self, host: str, port: int, timeout: float = 1.0) -> bool:
        """
        扫描指定端口是否开放
        
        Args:
            host: 目标主机
            port: 端口号
            timeout: 超时时间
            
        Returns:
            bool: 端口是否开放
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def detect_suspicious_activity(self) -> List[SecurityAlert]:
        """
        检测可疑活动
        
        Returns:
            List[SecurityAlert]: 安全警报列表
        """
        alerts = []
        
        try:
            # 检测端口扫描
            connections = psutil.net_connections(kind='inet')
            port_attempts = defaultdict(int)
            
            for conn in connections:
                if conn.raddr:
                    target_ip = conn.raddr.ip
                    target_port = conn.raddr.port
                    
                    # 检查是否对同一IP的多个端口进行连接
                    port_attempts[f"{target_ip}:{target_port}"] += 1
            
            # 简单的端口扫描检测
            for key, count in port_attempts.items():
                if count > 10:  # 阈值可配置
                    ip, port = key.split(":")
                    alert = SecurityAlert(
                        alert_id=hashlib.md5(f"{ip}:{port}:{time.time()}".encode()).hexdigest()[:8],
                        timestamp=datetime.now(),
                        severity="MEDIUM",
                        alert_type="端口扫描检测",
                        description=f"检测到对 {ip}:{port} 的频繁连接尝试 ({count} 次)",
                        source_ip=None,
                        target_ip=ip
                    )
                    alerts.append(alert)
            
            # 检测异常流量
            bandwidth_info = self.monitor_bandwidth_usage(0.1)
            for interface, data in bandwidth_info.get("interfaces", {}).items():
                if data["total_rate_mbps"] > 100:  # 100Mbps阈值
                    alert = SecurityAlert(
                        alert_id=hashlib.md5(f"{interface}:{time.time()}".encode()).hexdigest()[:8],
                        timestamp=datetime.now(),
                        severity="HIGH",
                        alert_type="异常高流量",
                        description=f"接口 {interface} 检测到异常高流量: {data['total_rate_mbps']:.2f} Mbps",
                        mitigation="检查是否有恶意流量或大量数据传输"
                    )
                    alerts.append(alert)
            
        except Exception as e:
            logger.error(f"可疑活动检测失败: {e}")
        
        return alerts
    
    def monitor_network_security(self) -> Dict[str, Any]:
        """
        监控网络安全状态
        
        Returns:
            Dict[str, Any]: 安全监控结果
        """
        security_info = {
            "open_ports": {},
            "suspicious_activities": [],
            "security_score": 100.0,
            "threats_detected": 0,
            "timestamp": datetime.now()
        }
        
        try:
            # 扫描常用端口
            common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3389, 5432, 3306]
            local_ip = self._get_local_ip()
            
            for port in common_ports:
                if self.scan_port(local_ip, port):
                    security_info["open_ports"][port] = "open"
            
            # 检测可疑活动
            alerts = self.detect_suspicious_activity()
            security_info["suspicious_activities"] = alerts
            security_info["threats_detected"] = len(alerts)
            
            # 计算安全分数
            if alerts:
                for alert in alerts:
                    if alert.severity == "CRITICAL":
                        security_info["security_score"] -= 30
                    elif alert.severity == "HIGH":
                        security_info["security_score"] -= 20
                    elif alert.severity == "MEDIUM":
                        security_info["security_score"] -= 10
                    else:
                        security_info["security_score"] -= 5
            
            security_info["security_score"] = max(0, security_info["security_score"])
            
        except Exception as e:
            logger.error(f"安全监控失败: {e}")
            security_info["error"] = str(e)
        
        return security_info
    
    def _get_local_ip(self) -> str:
        """获取本机IP地址"""
        try:
            # 连接到外部地址获取本机IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    # ==================== 网络拓扑发现 ====================
    
    def discover_network_topology(self) -> Dict[str, NetworkTopology]:
        """
        发现网络拓扑结构
        
        Returns:
            Dict[str, NetworkTopology]: 网络拓扑信息
        """
        topology = {}
        
        try:
            # 获取本地网络接口
            interfaces = psutil.net_if_addrs()
            for interface_name, addresses in interfaces.items():
                if interface_name.startswith(('lo', 'docker', 'veth')):
                    continue
                
                for addr in addresses:
                    if addr.family == socket.AF_INET:  # IPv4
                        ip = addr.address
                        netmask = addr.netmask
                        
                        # 跳过回环地址
                        if ip.startswith(('127.', '192.168.', '10.', '172.')):
                            try:
                                network = ip_network(f"{ip}/{netmask}", strict=False)
                                node = NetworkTopology(
                                    node_id=f"{interface_name}:{ip}",
                                    ip_address=ip,
                                    hostname=socket.gethostbyaddr(ip)[0],
                                    device_type="host",
                                    status="online"
                                )
                                topology[node.node_id] = node
                            except Exception as e:
                                logger.warning(f"处理接口 {interface_name} 失败: {e}")
            
            # 扫描网络中的其他设备（简化实现）
            for node_id, node in topology.items():
                try:
                    network = ip_network(f"{node.ip_address}/24", strict=False)
                    for ip in network.hosts():
                        if str(ip) != node.ip_address:
                            # 简单的设备发现
                            latency, success = self.ping_host(str(ip), timeout=1.0)
                            if success:
                                neighbor_node = NetworkTopology(
                                    node_id=f"device:{ip}",
                                    ip_address=str(ip),
                                    hostname=str(ip),
                                    device_type="unknown",
                                    status="online"
                                )
                                topology[neighbor_node.node_id] = neighbor_node
                                node.neighbors.append(neighbor_node.node_id)
                except Exception as e:
                    logger.warning(f"扫描网络 {node.ip_address} 失败: {e}")
            
        except Exception as e:
            logger.error(f"网络拓扑发现失败: {e}")
        
        return topology
    
    # ==================== 网络流量分析 ====================
    
    def analyze_network_traffic(self, duration: int = 60) -> Dict[str, TrafficAnalysis]:
        """
        分析网络流量
        
        Args:
            duration: 分析持续时间（秒）
            
        Returns:
            Dict[str, TrafficAnalysis]: 流量分析结果
        """
        traffic_data = defaultdict(lambda: {
            "bytes_sent": 0,
            "bytes_recv": 0,
            "packets_sent": 0,
            "packets_recv": 0,
            "connections": defaultdict(int)
        })
        
        start_time = time.time()
        
        try:
            # 收集流量数据
            while time.time() - start_time < duration:
                connections = self.get_active_connections()
                
                for conn in connections:
                    key = f"{conn.protocol}:{conn.dest_ip}"
                    
                    # 统计连接信息
                    traffic_data[key]["connections"][conn.source_ip] += 1
                    
                    # 这里可以添加更详细的流量统计
                    # 由于psutil限制，我们使用简化实现
                
                time.sleep(1)
            
            # 分析流量模式
            analysis_results = {}
            for key, data in traffic_data.items():
                if data["connections"]:
                    protocol, dest_ip = key.split(":", 1)
                    
                    # 计算流量模式
                    total_connections = sum(data["connections"].values())
                    unique_sources = len(data["connections"])
                    
                    if unique_sources > 10:
                        pattern = "高并发"
                    elif total_connections > 50:
                        pattern = "大量传输"
                    else:
                        pattern = "正常"
                    
                    analysis = TrafficAnalysis(
                        protocol=protocol,
                        source_ips=dict(data["connections"]),
                        dest_ips={dest_ip: total_connections},
                        total_bytes=0,  # 需要更详细的流量监控
                        packet_count=total_connections,
                        duration=duration,
                        avg_packet_size=0.0,
                        peak_bandwidth=0.0,
                        flow_pattern=pattern
                    )
                    analysis_results[key] = analysis
            
        except Exception as e:
            logger.error(f"流量分析失败: {e}")
        
        return analysis_results
    
    # ==================== 网络性能优化 ====================
    
    def analyze_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        分析性能瓶颈
        
        Returns:
            List[Dict[str, Any]]: 性能瓶颈分析结果
        """
        bottlenecks = []
        
        try:
            # CPU和内存使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            if cpu_percent > 80:
                bottlenecks.append({
                    "type": "CPU使用率过高",
                    "severity": "HIGH",
                    "value": f"{cpu_percent:.1f}%",
                    "recommendation": "考虑优化应用程序或增加CPU资源"
                })
            
            if memory.percent > 80:
                bottlenecks.append({
                    "type": "内存使用率过高",
                    "severity": "HIGH",
                    "value": f"{memory.percent:.1f}%",
                    "recommendation": "考虑增加内存或优化内存使用"
                })
            
            # 网络延迟
            latency_results = self.monitor_network_latency()
            for host, data in latency_results.items():
                if data["avg_latency"] > self.config["latency_threshold"]:
                    bottlenecks.append({
                        "type": "网络延迟过高",
                        "severity": "MEDIUM",
                        "target": host,
                        "value": f"{data['avg_latency']:.1f}ms",
                        "recommendation": "检查网络连接或优化路由"
                    })
            
            # 带宽使用率
            bandwidth_info = self.monitor_bandwidth_usage()
            for interface, data in bandwidth_info.get("interfaces", {}).items():
                if data["total_rate_mbps"] > 100:  # 假设100Mbps为高使用率
                    bottlenecks.append({
                        "type": "带宽使用率过高",
                        "severity": "MEDIUM",
                        "target": interface,
                        "value": f"{data['total_rate_mbps']:.1f}Mbps",
                        "recommendation": "优化网络流量或升级网络带宽"
                    })
            
            # 错误率
            error_info = self.monitor_network_errors()
            if error_info.get("total_errors", 0) > self.config["error_threshold"]:
                bottlenecks.append({
                    "type": "网络错误过多",
                    "severity": "HIGH",
                    "value": f"{error_info['total_errors']} 错误",
                    "recommendation": "检查网络硬件和配置"
                })
            
        except Exception as e:
            logger.error(f"性能瓶颈分析失败: {e}")
        
        return bottlenecks
    
    def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        生成性能优化建议
        
        Returns:
            List[Dict[str, Any]]: 优化建议列表
        """
        recommendations = []
        
        try:
            # 基于性能瓶颈生成建议
            bottlenecks = self.analyze_performance_bottlenecks()
            
            for bottleneck in bottlenecks:
                if bottleneck["type"] == "CPU使用率过高":
                    recommendations.append({
                        "category": "系统优化",
                        "priority": "HIGH",
                        "title": "CPU性能优化",
                        "description": "CPU使用率过高，建议：1) 优化应用程序算法 2) 使用多线程处理 3) 考虑升级CPU",
                        "impact": "高",
                        "effort": "中"
                    })
                
                elif bottleneck["type"] == "内存使用率过高":
                    recommendations.append({
                        "category": "系统优化",
                        "priority": "HIGH",
                        "title": "内存优化",
                        "description": "内存使用率过高，建议：1) 优化内存分配 2) 清理无用进程 3) 增加物理内存",
                        "impact": "高",
                        "effort": "中"
                    })
                
                elif bottleneck["type"] == "网络延迟过高":
                    recommendations.append({
                        "category": "网络优化",
                        "priority": "MEDIUM",
                        "title": "网络延迟优化",
                        "description": f"到 {bottleneck['target']} 的延迟过高，建议：1) 检查网络路径 2) 优化DNS解析 3) 使用CDN",
                        "impact": "中",
                        "effort": "低"
                    })
                
                elif bottleneck["type"] == "带宽使用率过高":
                    recommendations.append({
                        "category": "网络优化",
                        "priority": "MEDIUM",
                        "title": "带宽优化",
                        "description": f"接口 {bottleneck['target']} 带宽使用率过高，建议：1) 压缩数据传输 2) 使用QoS策略 3) 升级带宽",
                        "impact": "中",
                        "effort": "高"
                    })
            
            # 添加通用优化建议
            recommendations.extend([
                {
                    "category": "监控优化",
                    "priority": "LOW",
                    "title": "监控频率优化",
                    "description": "根据网络负载调整监控频率，在低负载时减少监控频率以节省资源",
                    "impact": "低",
                    "effort": "低"
                },
                {
                    "category": "安全优化",
                    "priority": "MEDIUM",
                    "title": "安全配置优化",
                    "description": "定期更新安全策略，配置防火墙规则，实施访问控制",
                    "impact": "中",
                    "effort": "中"
                }
            ])
            
        except Exception as e:
            logger.error(f"生成优化建议失败: {e}")
        
        return recommendations
    
    # ==================== 网络监控报告 ====================
    
    def generate_monitoring_report(self, duration_hours: int = 24) -> Dict[str, Any]:
        """
        生成监控报告
        
        Args:
            duration_hours: 报告时间范围（小时）
            
        Returns:
            Dict[str, Any]: 监控报告
        """
        report = {
            "report_info": {
                "generated_at": datetime.now(),
                "duration_hours": duration_hours,
                "report_version": "1.0.0"
            },
            "summary": {},
            "network_health": {},
            "performance_metrics": {},
            "security_status": {},
            "topology_info": {},
            "traffic_analysis": {},
            "optimization_recommendations": [],
            "alerts": []
        }
        
        try:
            # 网络健康状态
            health_info = self.monitor_connection_health()
            report["network_health"] = health_info
            
            # 性能指标
            latency_results = self.monitor_network_latency()
            bandwidth_info = self.monitor_bandwidth_usage()
            error_info = self.monitor_network_errors()
            
            report["performance_metrics"] = {
                "latency": latency_results,
                "bandwidth": bandwidth_info,
                "errors": error_info
            }
            
            # 安全状态
            security_info = self.monitor_network_security()
            report["security_status"] = security_info
            
            # 网络拓扑
            topology = self.discover_network_topology()
            report["topology_info"] = {
                "total_nodes": len(topology),
                "nodes": {node_id: {
                    "ip_address": node.ip_address,
                    "hostname": node.hostname,
                    "device_type": node.device_type,
                    "status": node.status
                } for node_id, node in topology.items()}
            }
            
            # 流量分析
            traffic_analysis = self.analyze_network_traffic(60)  # 1分钟分析
            report["traffic_analysis"] = {
                "protocols": {key: {
                    "protocol": analysis.protocol,
                    "unique_sources": len(analysis.source_ips),
                    "flow_pattern": analysis.flow_pattern
                } for key, analysis in traffic_analysis.items()}
            }
            
            # 优化建议
            recommendations = self.generate_optimization_recommendations()
            report["optimization_recommendations"] = recommendations
            
            # 安全警报
            alerts = [alert for alert in self.security_alerts 
                     if alert.timestamp > datetime.now() - timedelta(hours=duration_hours)]
            report["alerts"] = [{
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp,
                "severity": alert.severity,
                "type": alert.alert_type,
                "description": alert.description
            } for alert in alerts]
            
            # 汇总信息
            report["summary"] = {
                "total_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a.severity == "CRITICAL"]),
                "high_alerts": len([a for a in alerts if a.severity == "HIGH"]),
                "network_status": "healthy" if len(alerts) == 0 else "warning" if len([a for a in alerts if a.severity in ["HIGH", "CRITICAL"]]) == 0 else "critical",
                "avg_latency": statistics.mean([data["avg_latency"] for data in latency_results.values()]) if latency_results else 0,
                "total_bandwidth_usage": bandwidth_info.get("total_bandwidth_usage", 0),
                "security_score": security_info.get("security_score", 100)
            }
            
        except Exception as e:
            logger.error(f"生成监控报告失败: {e}")
            report["error"] = str(e)
        
        return report
    
    def export_report(self, report: Dict[str, Any], format: str = "json", 
                     filename: Optional[str] = None) -> str:
        """
        导出监控报告
        
        Args:
            report: 监控报告数据
            format: 导出格式 ("json", "txt", "html")
            filename: 文件名（可选）
            
        Returns:
            str: 导出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"network_monitor_report_{timestamp}"
        
        try:
            if format.lower() == "json":
                filepath = f"/workspace/D/AO/AOO/M/M4/{filename}.json"
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            elif format.lower() == "txt":
                filepath = f"/workspace/D/AO/AOO/M/M4/{filename}.txt"
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("网络监控报告\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # 基本信息
                    f.write(f"生成时间: {report['report_info']['generated_at']}\n")
                    f.write(f"监控时长: {report['report_info']['duration_hours']} 小时\n\n")
                    
                    # 汇总信息
                    summary = report.get("summary", {})
                    f.write("汇总信息:\n")
                    f.write("-" * 20 + "\n")
                    for key, value in summary.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                    
                    # 性能指标
                    f.write("性能指标:\n")
                    f.write("-" * 20 + "\n")
                    metrics = report.get("performance_metrics", {})
                    if metrics.get("latency"):
                        f.write("延迟监控:\n")
                        for host, data in metrics["latency"].items():
                            f.write(f"  {host}: 平均延迟 {data['avg_latency']:.2f}ms\n")
                    f.write("\n")
                    
                    # 安全状态
                    f.write("安全状态:\n")
                    f.write("-" * 20 + "\n")
                    security = report.get("security_status", {})
                    f.write(f"安全分数: {security.get('security_score', 'N/A')}\n")
                    f.write(f"检测到的威胁: {security.get('threats_detected', 0)}\n")
                    f.write("\n")
                    
                    # 优化建议
                    recommendations = report.get("optimization_recommendations", [])
                    if recommendations:
                        f.write("优化建议:\n")
                        f.write("-" * 20 + "\n")
                        for i, rec in enumerate(recommendations, 1):
                            f.write(f"{i}. {rec['title']} (优先级: {rec['priority']})\n")
                            f.write(f"   {rec['description']}\n\n")
            
            elif format.lower() == "html":
                filepath = f"/workspace/D/AO/AOO/M/M4/{filename}.html"
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>网络监控报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metric { margin: 10px 0; }
        .alert { padding: 10px; margin: 5px 0; border-radius: 3px; }
        .critical { background-color: #ffebee; border-left: 4px solid #f44336; }
        .high { background-color: #fff3e0; border-left: 4px solid #ff9800; }
        .medium { background-color: #fff8e1; border-left: 4px solid #ffc107; }
        .low { background-color: #f1f8e9; border-left: 4px solid #8bc34a; }
    </style>
</head>
<body>
    <div class="header">
        <h1>网络监控报告</h1>
        <p>生成时间: """ + str(report['report_info']['generated_at']) + """</p>
        <p>监控时长: """ + str(report['report_info']['duration_hours']) + """ 小时</p>
    </div>
""")
                    
                    # 汇总信息
                    summary = report.get("summary", {})
                    f.write(f"""
    <div class="section">
        <h2>汇总信息</h2>
        <div class="metric">网络状态: {summary.get('network_status', 'unknown')}</div>
        <div class="metric">总警报数: {summary.get('total_alerts', 0)}</div>
        <div class="metric">严重警报: {summary.get('critical_alerts', 0)}</div>
        <div class="metric">平均延迟: {summary.get('avg_latency', 0):.2f}ms</div>
        <div class="metric">安全分数: {summary.get('security_score', 100)}</div>
    </div>
""")
                    
                    # 警报列表
                    alerts = report.get("alerts", [])
                    if alerts:
                        f.write('    <div class="section">\n        <h2>安全警报</h2>\n')
                        for alert in alerts:
                            f.write(f'        <div class="alert {alert["severity"].lower()}">')
                            f.write(f'<strong>{alert["type"]}</strong> - {alert["description"]} ')
                            f.write(f'({alert["timestamp"]})</div>\n')
                        f.write('    </div>\n')
                    
                    f.write("</body></html>")
            
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logger.info(f"报告已导出到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"导出报告失败: {e}")
            raise
    
    # ==================== 监控控制 ====================
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            logger.warning("监控已在运行中")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("网络监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("网络监控已停止")
    
    def _monitoring_loop(self):
        """监控主循环"""
        while self.is_monitoring:
            try:
                # 收集指标
                metrics = self._collect_metrics()
                with self.data_lock:
                    self.metrics_history.append(metrics)
                
                # 执行回调
                for callback in self.metrics_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"指标回调执行失败: {e}")
                
                # 检查安全
                if self.config.get("enable_security_monitoring", True):
                    alerts = self.detect_suspicious_activity()
                    with self.data_lock:
                        self.security_alerts.extend(alerts)
                    
                    for alert in alerts:
                        for callback in self.alert_callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                logger.error(f"警报回调执行失败: {e}")
                
                time.sleep(self.config["monitor_interval"])
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(5)
    
    def _collect_metrics(self) -> NetworkMetrics:
        """收集网络指标"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            network_io = psutil.net_io_counters()
            connections = psutil.net_connections(kind='inet')
            
            # 计算带宽使用率（简化实现）
            bandwidth_usage = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)  # MB
            
            # 计算延迟（简化实现）
            latency = 0.0
            try:
                latency_result = self.ping_host("8.8.8.8")
                latency = latency_result[0] if latency_result[1] else 0.0
            except Exception:
                pass
            
            # 计算丢包率（简化实现）
            packet_loss = 0.0
            
            # 统计错误
            errors_count = sum([
                network_io.errin,
                network_io.errout,
                network_io.dropin,
                network_io.dropout
            ])
            
            return NetworkMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                network_io={
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                },
                connections_count=len(connections),
                bandwidth_usage=bandwidth_usage,
                latency=latency,
                packet_loss=packet_loss,
                errors_count=errors_count
            )
            
        except Exception as e:
            logger.error(f"收集指标失败: {e}")
            raise
    
    # ==================== 回调管理 ====================
    
    def add_alert_callback(self, callback: Callable[[SecurityAlert], None]):
        """添加警报回调函数"""
        self.alert_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable[[NetworkMetrics], None]):
        """添加指标回调函数"""
        self.metrics_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[SecurityAlert], None]):
        """移除警报回调函数"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def remove_metrics_callback(self, callback: Callable[[NetworkMetrics], None]):
        """移除指标回调函数"""
        if callback in self.metrics_callbacks:
            self.metrics_callbacks.remove(callback)
    
    # ==================== 配置管理 ====================
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        with self.data_lock:
            self.config.update(new_config)
        logger.info("配置已更新")
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()
    
    def reset_to_default_config(self):
        """重置为默认配置"""
        with self.data_lock:
            self.config = self._default_config()
        logger.info("配置已重置为默认值")


# ==================== 测试用例 ====================

def test_network_monitor():
    """网络监控器测试函数"""
    print("开始测试网络监控器...")
    
    # 创建监控器实例
    monitor = NetworkMonitor()
    
    # 测试基本功能
    print("\n1. 测试网络连接监控")
    connections = monitor.get_active_connections()
    print(f"活跃连接数: {len(connections)}")
    
    health_info = monitor.monitor_connection_health()
    print(f"连接健康状态: {health_info}")
    
    print("\n2. 测试网络延迟监控")
    latency_results = monitor.monitor_network_latency(["8.8.8.8", "114.114.114.114"])
    for host, data in latency_results.items():
        print(f"{host}: 平均延迟 {data['avg_latency']:.2f}ms, 丢包率 {data['packet_loss']:.1f}%")
    
    print("\n3. 测试带宽使用监控")
    bandwidth_info = monitor.monitor_bandwidth_usage()
    print(f"总带宽使用: {bandwidth_info['total_bandwidth_usage']:.2f} Mbps")
    
    print("\n4. 测试网络错误监控")
    error_info = monitor.monitor_network_errors()
    print(f"总错误数: {error_info['total_errors']}")
    
    print("\n5. 测试网络安全监控")
    security_info = monitor.monitor_network_security()
    print(f"安全分数: {security_info['security_score']}")
    print(f"检测到的威胁: {security_info['threats_detected']}")
    
    print("\n6. 测试网络拓扑发现")
    topology = monitor.discover_network_topology()
    print(f"发现的节点数: {len(topology)}")
    
    print("\n7. 测试流量分析")
    traffic_analysis = monitor.analyze_network_traffic(10)  # 10秒分析
    print(f"分析的流量类型数: {len(traffic_analysis)}")
    
    print("\n8. 测试性能优化分析")
    bottlenecks = monitor.analyze_performance_bottlenecks()
    print(f"发现的性能瓶颈: {len(bottlenecks)}")
    for bottleneck in bottlenecks:
        print(f"  - {bottleneck['type']}: {bottleneck['value']}")
    
    recommendations = monitor.generate_optimization_recommendations()
    print(f"生成的优化建议: {len(recommendations)}")
    
    print("\n9. 测试监控报告生成")
    report = monitor.generate_monitoring_report(1)  # 1小时报告
    print(f"报告摘要: {report['summary']}")
    
    print("\n10. 测试报告导出")
    try:
        json_file = monitor.export_report(report, "json", "test_report")
        print(f"JSON报告已导出: {json_file}")
        
        txt_file = monitor.export_report(report, "txt", "test_report")
        print(f"TXT报告已导出: {txt_file}")
        
        html_file = monitor.export_report(report, "html", "test_report")
        print(f"HTML报告已导出: {html_file}")
    except Exception as e:
        print(f"报告导出失败: {e}")
    
    print("\n11. 测试监控控制")
    print("启动监控...")
    monitor.start_monitoring()
    time.sleep(5)  # 运行5秒
    
    print("停止监控...")
    monitor.stop_monitoring()
    
    print("\n网络监控器测试完成！")


def demo_alert_callback(alert: SecurityAlert):
    """演示警报回调函数"""
    print(f"[ALERT] {alert.severity} - {alert.alert_type}: {alert.description}")


def demo_metrics_callback(metrics: NetworkMetrics):
    """演示指标回调函数"""
    print(f"[METRICS] CPU: {metrics.cpu_percent:.1f}%, 内存: {metrics.memory_percent:.1f}%, "
          f"连接数: {metrics.connections_count}")


def demo_monitoring_with_callbacks():
    """演示带回调的监控"""
    print("开始演示带回调的监控...")
    
    monitor = NetworkMonitor()
    
    # 添加回调函数
    monitor.add_alert_callback(demo_alert_callback)
    monitor.add_metrics_callback(demo_metrics_callback)
    
    # 启动监控
    monitor.start_monitoring()
    
    try:
        # 运行30秒监控
        print("监控运行中，等待30秒...")
        time.sleep(30)
    except KeyboardInterrupt:
        print("用户中断监控")
    finally:
        monitor.stop_monitoring()
        print("监控已停止")


if __name__ == "__main__":
    # 运行测试
    test_network_monitor()
    
    print("\n" + "="*50)
    
    # 运行演示
    try:
        demo_monitoring_with_callbacks()
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        print("这可能是由于权限限制或网络环境问题导致的，属于正常现象。")