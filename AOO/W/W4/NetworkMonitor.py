#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W4网络监控器 - 主要网络监控实现
提供网络流量监控、性能监控、故障检测等功能
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
import socket
import psutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import ping3
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class NetworkMetrics:
    """网络指标数据类"""
    timestamp: datetime
    interface: str
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    errors_in: int
    errors_out: int
    drops_in: int
    drops_out: int


@dataclass
class NetworkPerformance:
    """网络性能数据类"""
    timestamp: datetime
    target_host: str
    latency: float
    packet_loss: float
    bandwidth: float
    jitter: float


@dataclass
class NetworkAlert:
    """网络告警数据类"""
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    source: str
    resolved: bool = False


class NetworkMonitor:
    """W4网络监控器主类"""
    
    def __init__(self, config_file: str = "network_monitor_config.json"):
        """初始化网络监控器
        
        Args:
            config_file: 配置文件路径
        """
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # 监控数据存储
        self.metrics_history = defaultdict(deque)
        self.performance_history = defaultdict(deque)
        self.alerts_history = deque(maxlen=1000)
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_threads = {}
        
        # 网络接口
        self.interfaces = self._get_network_interfaces()
        
        # 告警阈值配置
        self.thresholds = {
            'latency_high': 100,  # ms
            'packet_loss_high': 5,  # %
            'bandwidth_low': 1024 * 1024,  # 1MB/s
            'cpu_usage_high': 80,  # %
            'memory_usage_high': 80  # %
        }
        
        self.logger.info("W4网络监控器初始化完成")
    
    def _load_config(self, config_file: str) -> Dict:
        """加载配置文件"""
        default_config = {
            "monitoring_interval": 5,  # 监控间隔（秒）
            "history_size": 1000,  # 历史数据大小
            "targets": [
                "8.8.8.8",
                "114.114.114.114",
                "baidu.com"
            ],
            "alert_settings": {
                "enable_email_alerts": False,
                "enable_webhook_alerts": False,
                "email_recipients": [],
                "webhook_url": ""
            },
            "report_settings": {
                "auto_generate": True,
                "report_interval": 3600,  # 1小时
                "output_directory": "reports"
            }
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 合并默认配置
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            else:
                # 创建默认配置文件
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                return default_config
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
            return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志记录"""
        logger = logging.getLogger("W4NetworkMonitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 文件处理器
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler("logs/network_monitor.log", encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def _get_network_interfaces(self) -> List[str]:
        """获取网络接口列表"""
        interfaces = []
        try:
            for interface, addrs in psutil.net_if_addrs().items():
                if interface != 'lo':  # 排除本地回环接口
                    interfaces.append(interface)
        except Exception as e:
            self.logger.error(f"获取网络接口失败: {e}")
        return interfaces
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            self.logger.warning("监控已在运行中")
            return
        
        self.is_monitoring = True
        self.logger.info("开始网络监控")
        
        # 启动各种监控线程
        self.monitor_threads['traffic'] = threading.Thread(
            target=self._monitor_network_traffic, daemon=True
        )
        self.monitor_threads['performance'] = threading.Thread(
            target=self._monitor_network_performance, daemon=True
        )
        self.monitor_threads['fault_detection'] = threading.Thread(
            target=self._monitor_network_faults, daemon=True
        )
        self.monitor_threads['topology'] = threading.Thread(
            target=self._monitor_network_topology, daemon=True
        )
        self.monitor_threads['statistics'] = threading.Thread(
            target=self._monitor_network_statistics, daemon=True
        )
        self.monitor_threads['alerts'] = threading.Thread(
            target=self._monitor_network_alerts, daemon=True
        )
        
        # 启动所有线程
        for thread in self.monitor_threads.values():
            thread.start()
        
        # 启动报告生成器
        if self.config.get('report_settings', {}).get('auto_generate', True):
            self.monitor_threads['reports'] = threading.Thread(
                target=self._auto_generate_reports, daemon=True
            )
            self.monitor_threads['reports'].start()
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            self.logger.warning("监控未在运行")
            return
        
        self.is_monitoring = False
        self.logger.info("停止网络监控")
        
        # 等待所有线程结束
        for thread in self.monitor_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.monitor_threads.clear()
    
    def _monitor_network_traffic(self):
        """网络流量监控"""
        previous_stats = {}
        
        while self.is_monitoring:
            try:
                current_time = datetime.now()
                net_stats = psutil.net_io_counters(pernic=True)
                
                for interface, stats in net_stats.items():
                    if interface in self.interfaces:
                        metric = NetworkMetrics(
                            timestamp=current_time,
                            interface=interface,
                            bytes_sent=stats.bytes_sent,
                            bytes_recv=stats.bytes_recv,
                            packets_sent=stats.packets_sent,
                            packets_recv=stats.packets_recv,
                            errors_in=stats.errin,
                            errors_out=stats.errout,
                            drops_in=stats.dropin,
                            drops_out=stats.dropout
                        )
                        
                        # 计算流量速率
                        if interface in previous_stats:
                            prev_stats = previous_stats[interface]
                            time_diff = (current_time - prev_stats.timestamp).total_seconds()
                            
                            if time_diff > 0:
                                bytes_sent_rate = (metric.bytes_sent - prev_stats.bytes_sent) / time_diff
                                bytes_recv_rate = (metric.bytes_recv - prev_stats.bytes_recv) / time_diff
                                
                                # 检查流量异常
                                if bytes_sent_rate > 100 * 1024 * 1024:  # 100MB/s
                                    self._create_alert(
                                        "high_traffic", "warning",
                                        f"接口 {interface} 出站流量过高: {bytes_sent_rate / 1024 / 1024:.2f} MB/s",
                                        "traffic_monitor"
                                    )
                        
                        # 保存指标
                        self.metrics_history[interface].append(metric)
                        
                        # 限制历史数据大小
                        max_history = self.config.get('history_size', 1000)
                        if len(self.metrics_history[interface]) > max_history:
                            self.metrics_history[interface].popleft()
                        
                        previous_stats[interface] = metric
                
                time.sleep(self.config.get('monitoring_interval', 5))
                
            except Exception as e:
                self.logger.error(f"网络流量监控错误: {e}")
                time.sleep(5)
    
    def _monitor_network_performance(self):
        """网络性能监控"""
        targets = self.config.get('targets', ['8.8.8.8'])
        
        while self.is_monitoring:
            try:
                current_time = datetime.now()
                
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_target = {
                        executor.submit(self._measure_network_performance, target): target
                        for target in targets
                    }
                    
                    for future in as_completed(future_to_target):
                        target = future_to_target[future]
                        try:
                            performance = future.result()
                            if performance:
                                self.performance_history[target].append(performance)
                                
                                # 检查性能异常
                                if performance.latency > self.thresholds['latency_high']:
                                    self._create_alert(
                                        "high_latency", "warning",
                                        f"到 {target} 的延迟过高: {performance.latency:.2f}ms",
                                        "performance_monitor"
                                    )
                                
                                if performance.packet_loss > self.thresholds['packet_loss_high']:
                                    self._create_alert(
                                        "packet_loss", "critical",
                                        f"到 {target} 的丢包率过高: {performance.packet_loss:.2f}%",
                                        "performance_monitor"
                                    )
                                
                                # 限制历史数据大小
                                max_history = self.config.get('history_size', 1000)
                                if len(self.performance_history[target]) > max_history:
                                    self.performance_history[target].popleft()
                        
                        except Exception as e:
                            self.logger.error(f"测量到 {target} 的网络性能失败: {e}")
                
                time.sleep(self.config.get('monitoring_interval', 5))
                
            except Exception as e:
                self.logger.error(f"网络性能监控错误: {e}")
                time.sleep(5)
    
    def _measure_network_performance(self, target: str) -> Optional[NetworkPerformance]:
        """测量网络性能指标"""
        try:
            current_time = datetime.now()
            
            # 测量延迟
            latency_results = []
            packet_loss_count = 0
            total_pings = 5
            
            for _ in range(total_pings):
                try:
                    latency = ping3.ping(target, timeout=2)
                    if latency is None:
                        packet_loss_count += 1
                    else:
                        latency_results.append(latency * 1000)  # 转换为毫秒
                except Exception:
                    packet_loss_count += 1
                
                time.sleep(0.1)
            
            # 计算平均延迟和抖动
            avg_latency = statistics.mean(latency_results) if latency_results else 0
            packet_loss = (packet_loss_count / total_pings) * 100
            jitter = statistics.stdev(latency_results) if len(latency_results) > 1 else 0
            
            # 测量带宽（简化版本）
            bandwidth = self._measure_bandwidth(target)
            
            return NetworkPerformance(
                timestamp=current_time,
                target_host=target,
                latency=avg_latency,
                packet_loss=packet_loss,
                bandwidth=bandwidth,
                jitter=jitter
            )
        
        except Exception as e:
            self.logger.error(f"测量网络性能失败 {target}: {e}")
            return None
    
    def _measure_bandwidth(self, target: str) -> float:
        """测量带宽（简化实现）"""
        try:
            # 这里使用简单的HTTP请求来估算带宽
            start_time = time.time()
            try:
                response = requests.get(f"http://{target}", timeout=5, stream=True)
                total_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    total_size += len(chunk)
                    if time.time() - start_time > 3:  # 限制测试时间
                        break
                
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    return total_size / elapsed_time  # bytes per second
            except:
                pass
            
            return 0.0
        
        except Exception:
            return 0.0
    
    def _monitor_network_faults(self):
        """网络故障检测"""
        previous_status = {}
        
        while self.is_monitoring:
            try:
                current_time = datetime.now()
                
                # 检查网络连接状态
                for target in self.config.get('targets', []):
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(3)
                        result = sock.connect_ex((target, 80))
                        sock.close()
                        
                        is_connected = result == 0
                        
                        # 检查状态变化
                        if target in previous_status:
                            if previous_status[target] and not is_connected:
                                self._create_alert(
                                    "connection_lost", "critical",
                                    f"与 {target} 的连接丢失",
                                    "fault_detector"
                                )
                            elif not previous_status[target] and is_connected:
                                self._create_alert(
                                    "connection_restored", "info",
                                    f"与 {target} 的连接已恢复",
                                    "fault_detector"
                                )
                        
                        previous_status[target] = is_connected
                    
                    except Exception as e:
                        self.logger.error(f"检查到 {target} 的连接状态失败: {e}")
                
                time.sleep(10)  # 故障检测间隔稍长一些
                
            except Exception as e:
                self.logger.error(f"网络故障检测错误: {e}")
                time.sleep(10)
    
    def _monitor_network_topology(self):
        """网络拓扑监控"""
        while self.is_monitoring:
            try:
                # 获取网络路由表
                routes = psutil.net_connections(kind='inet')
                active_connections = []
                
                for conn in routes:
                    if conn.status == 'ESTABLISHED':
                        active_connections.append({
                            'local_address': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "Unknown",
                            'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "Unknown",
                            'pid': conn.pid,
                            'status': conn.status
                        })
                
                # 这里可以添加更复杂的拓扑发现逻辑
                # 例如：ARP表扫描、路由跟踪等
                
                self.logger.debug(f"发现 {len(active_connections)} 个活动连接")
                
                time.sleep(30)  # 拓扑监控间隔较长
                
            except Exception as e:
                self.logger.error(f"网络拓扑监控错误: {e}")
                time.sleep(30)
    
    def _monitor_network_statistics(self):
        """网络统计监控"""
        while self.is_monitoring:
            try:
                # CPU和内存使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # 检查系统资源使用情况
                if cpu_percent > self.thresholds['cpu_usage_high']:
                    self._create_alert(
                        "high_cpu_usage", "warning",
                        f"CPU使用率过高: {cpu_percent:.1f}%",
                        "statistics_monitor"
                    )
                
                if memory.percent > self.thresholds['memory_usage_high']:
                    self._create_alert(
                        "high_memory_usage", "warning",
                        f"内存使用率过高: {memory.percent:.1f}%",
                        "statistics_monitor"
                    )
                
                # 网络连接数统计
                connection_count = len(psutil.net_connections())
                if connection_count > 1000:
                    self._create_alert(
                        "high_connection_count", "info",
                        f"网络连接数过高: {connection_count}",
                        "statistics_monitor"
                    )
                
                time.sleep(60)  # 统计监控间隔
                
            except Exception as e:
                self.logger.error(f"网络统计监控错误: {e}")
                time.sleep(60)
    
    def _monitor_network_alerts(self):
        """网络告警处理"""
        while self.is_monitoring:
            try:
                # 处理未解决的告警
                for alert in list(self.alerts_history):
                    if not alert.resolved:
                        # 检查是否应该自动解决告警
                        if self._should_resolve_alert(alert):
                            alert.resolved = True
                            self.logger.info(f"告警已自动解决: {alert.message}")
                
                # 发送告警通知
                self._send_alert_notifications()
                
                time.sleep(30)  # 告警处理间隔
                
            except Exception as e:
                self.logger.error(f"网络告警处理错误: {e}")
                time.sleep(30)
    
    def _should_resolve_alert(self, alert: NetworkAlert) -> bool:
        """判断告警是否应该自动解决"""
        try:
            # 根据告警类型和时间判断
            alert_age = (datetime.now() - alert.timestamp).total_seconds()
            
            if alert_age > 300:  # 5分钟后自动解决
                return True
            
            # 对于性能告警，检查当前状态是否正常
            if alert.alert_type in ['high_latency', 'packet_loss']:
                target = alert.source.split('_')[-1] if '_' in alert.source else None
                if target and target in self.performance_history:
                    latest_perf = list(self.performance_history[target])[-1]
                    if alert.alert_type == 'high_latency' and latest_perf.latency < self.thresholds['latency_high'] * 0.8:
                        return True
                    elif alert.alert_type == 'packet_loss' and latest_perf.packet_loss < self.thresholds['packet_loss_high'] * 0.5:
                        return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"判断告警解决状态失败: {e}")
            return False
    
    def _send_alert_notifications(self):
        """发送告警通知"""
        try:
            # 获取最近的未解决告警
            recent_alerts = [
                alert for alert in self.alerts_history
                if not alert.resolved and 
                (datetime.now() - alert.timestamp).total_seconds() < 300
            ]
            
            if not recent_alerts:
                return
            
            # 邮件通知
            if self.config.get('alert_settings', {}).get('enable_email_alerts', False):
                self._send_email_alerts(recent_alerts)
            
            # Webhook通知
            if self.config.get('alert_settings', {}).get('enable_webhook_alerts', False):
                self._send_webhook_alerts(recent_alerts)
        
        except Exception as e:
            self.logger.error(f"发送告警通知失败: {e}")
    
    def _send_email_alerts(self, alerts: List[NetworkAlert]):
        """发送邮件告警"""
        # 这里需要集成邮件发送功能
        # 可以使用smtplib或第三方邮件服务
        self.logger.info(f"准备发送 {len(alerts)} 个邮件告警")
    
    def _send_webhook_alerts(self, alerts: List[NetworkAlert]):
        """发送Webhook告警"""
        try:
            webhook_url = self.config.get('alert_settings', {}).get('webhook_url')
            if not webhook_url:
                return
            
            payload = {
                'timestamp': datetime.now().isoformat(),
                'alerts': [asdict(alert) for alert in alerts]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                self.logger.info("Webhook告警发送成功")
            else:
                self.logger.error(f"Webhook告警发送失败: {response.status_code}")
        
        except Exception as e:
            self.logger.error(f"发送Webhook告警失败: {e}")
    
    def _auto_generate_reports(self):
        """自动生成报告"""
        while self.is_monitoring:
            try:
                report_interval = self.config.get('report_settings', {}).get('report_interval', 3600)
                time.sleep(report_interval)
                
                if self.is_monitoring:
                    self.generate_network_report()
            
            except Exception as e:
                self.logger.error(f"自动生成报告失败: {e}")
                time.sleep(60)
    
    def _create_alert(self, alert_type: str, severity: str, message: str, source: str):
        """创建告警"""
        alert = NetworkAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            source=source
        )
        
        self.alerts_history.append(alert)
        self.logger.warning(f"[{severity.upper()}] {message}")
    
    def get_network_traffic_stats(self, interface: str = None, duration_minutes: int = 60) -> Dict:
        """获取网络流量统计"""
        try:
            if interface and interface in self.metrics_history:
                history = list(self.metrics_history[interface])
            else:
                # 合并所有接口的数据
                history = []
                for interface_history in self.metrics_history.values():
                    history.extend(list(interface_history))
            
            if not history:
                return {}
            
            # 过滤指定时间范围内的数据
            cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
            recent_history = [h for h in history if h.timestamp >= cutoff_time]
            
            if not recent_history:
                return {}
            
            # 计算统计信息
            total_bytes_sent = sum(h.bytes_sent for h in recent_history)
            total_bytes_recv = sum(h.bytes_recv for h in recent_history)
            total_packets_sent = sum(h.packets_sent for h in recent_history)
            total_packets_recv = sum(h.packets_recv for h in recent_history)
            total_errors = sum(h.errors_in + h.errors_out for h in recent_history)
            total_drops = sum(h.drops_in + h.drops_out for h in recent_history)
            
            # 计算平均流量速率
            time_span = (recent_history[-1].timestamp - recent_history[0].timestamp).total_seconds()
            if time_span > 0:
                avg_sent_rate = total_bytes_sent / time_span
                avg_recv_rate = total_bytes_recv / time_span
            else:
                avg_sent_rate = avg_recv_rate = 0
            
            return {
                'interface': interface or 'all',
                'duration_minutes': duration_minutes,
                'total_bytes_sent': total_bytes_sent,
                'total_bytes_recv': total_bytes_recv,
                'total_packets_sent': total_packets_sent,
                'total_packets_recv': total_packets_recv,
                'total_errors': total_errors,
                'total_drops': total_drops,
                'avg_sent_rate_bps': avg_sent_rate,
                'avg_recv_rate_bps': avg_recv_rate,
                'avg_sent_rate_mbps': avg_sent_rate / 1024 / 1024,
                'avg_recv_rate_mbps': avg_recv_rate / 1024 / 1024,
                'data_points': len(recent_history)
            }
        
        except Exception as e:
            self.logger.error(f"获取网络流量统计失败: {e}")
            return {}
    
    def get_network_performance_stats(self, target: str = None, duration_minutes: int = 60) -> Dict:
        """获取网络性能统计"""
        try:
            if target and target in self.performance_history:
                history = list(self.performance_history[target])
            else:
                # 合并所有目标的数据
                history = []
                for target_history in self.performance_history.values():
                    history.extend(list(target_history))
            
            if not history:
                return {}
            
            # 过滤指定时间范围内的数据
            cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
            recent_history = [h for h in history if h.timestamp >= cutoff_time]
            
            if not recent_history:
                return {}
            
            # 计算统计信息
            latencies = [h.latency for h in recent_history if h.latency > 0]
            packet_losses = [h.packet_loss for h in recent_history if h.packet_loss >= 0]
            bandwidths = [h.bandwidth for h in recent_history if h.bandwidth > 0]
            jitters = [h.jitter for h in recent_history if h.jitter >= 0]
            
            return {
                'target': target or 'all',
                'duration_minutes': duration_minutes,
                'avg_latency_ms': statistics.mean(latencies) if latencies else 0,
                'min_latency_ms': min(latencies) if latencies else 0,
                'max_latency_ms': max(latencies) if latencies else 0,
                'avg_packet_loss_percent': statistics.mean(packet_losses) if packet_losses else 0,
                'max_packet_loss_percent': max(packet_losses) if packet_losses else 0,
                'avg_bandwidth_bps': statistics.mean(bandwidths) if bandwidths else 0,
                'avg_bandwidth_mbps': statistics.mean(bandwidths) / 1024 / 1024 if bandwidths else 0,
                'avg_jitter_ms': statistics.mean(jitters) if jitters else 0,
                'data_points': len(recent_history)
            }
        
        except Exception as e:
            self.logger.error(f"获取网络性能统计失败: {e}")
            return {}
    
    def get_network_alerts(self, severity: str = None, limit: int = 50) -> List[Dict]:
        """获取网络告警"""
        try:
            alerts = list(self.alerts_history)
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            # 按时间倒序排列
            alerts.sort(key=lambda x: x.timestamp, reverse=True)
            
            return [asdict(alert) for alert in alerts[:limit]]
        
        except Exception as e:
            self.logger.error(f"获取网络告警失败: {e}")
            return []
    
    def get_network_topology(self) -> Dict:
        """获取网络拓扑信息"""
        try:
            # 获取网络接口信息
            interfaces_info = {}
            for interface, addrs in psutil.net_if_addrs().items():
                interfaces_info[interface] = []
                for addr in addrs:
                    interfaces_info[interface].append({
                        'family': str(addr.family),
                        'address': addr.address,
                        'netmask': addr.netmask,
                        'broadcast': addr.broadcast
                    })
            
            # 获取活动连接
            connections = []
            for conn in psutil.net_connections(kind='inet'):
                if conn.laddr and conn.raddr:
                    connections.append({
                        'local_address': f"{conn.laddr.ip}:{conn.laddr.port}",
                        'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}",
                        'status': conn.status,
                        'pid': conn.pid
                    })
            
            return {
                'interfaces': interfaces_info,
                'active_connections': connections,
                'total_connections': len(connections),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"获取网络拓扑失败: {e}")
            return {}
    
    def generate_network_report(self, output_file: str = None) -> str:
        """生成网络状态报告"""
        try:
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = self.config.get('report_settings', {}).get('output_directory', 'reports')
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/network_report_{timestamp}.json"
            
            # 收集所有监控数据
            report_data = {
                'report_info': {
                    'generated_at': datetime.now().isoformat(),
                    'monitor_version': 'W4 Network Monitor v1.0',
                    'report_type': 'network_status'
                },
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total': psutil.virtual_memory().total,
                    'disk_usage': psutil.disk_usage('/').percent,
                    'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
                },
                'network_interfaces': self.interfaces,
                'traffic_stats': {},
                'performance_stats': {},
                'alerts_summary': {},
                'topology_info': self.get_network_topology(),
                'recommendations': self._generate_recommendations()
            }
            
            # 流量统计
            for interface in self.interfaces:
                report_data['traffic_stats'][interface] = self.get_network_traffic_stats(interface, 60)
            
            # 性能统计
            for target in self.performance_history:
                report_data['performance_stats'][target] = self.get_network_performance_stats(target, 60)
            
            # 告警摘要
            all_alerts = self.get_network_alerts()
            critical_alerts = [a for a in all_alerts if a['severity'] == 'critical']
            warning_alerts = [a for a in all_alerts if a['severity'] == 'warning']
            
            report_data['alerts_summary'] = {
                'total_alerts': len(all_alerts),
                'critical_alerts': len(critical_alerts),
                'warning_alerts': len(warning_alerts),
                'recent_alerts': all_alerts[:10]
            }
            
            # 保存报告
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"网络报告已生成: {output_file}")
            return output_file
        
        except Exception as e:
            self.logger.error(f"生成网络报告失败: {e}")
            return ""
    
    def _generate_recommendations(self) -> List[str]:
        """生成网络优化建议"""
        recommendations = []
        
        try:
            # 基于性能数据生成建议
            for target, history in self.performance_history.items():
                if not history:
                    continue
                
                recent_perf = list(history)[-10:]  # 最近10次测量
                avg_latency = statistics.mean([p.latency for p in recent_perf if p.latency > 0])
                avg_packet_loss = statistics.mean([p.packet_loss for p in recent_perf])
                
                if avg_latency > 100:
                    recommendations.append(f"到 {target} 的延迟较高 ({avg_latency:.1f}ms)，建议检查网络路径或更换DNS服务器")
                
                if avg_packet_loss > 5:
                    recommendations.append(f"到 {target} 的丢包率较高 ({avg_packet_loss:.1f}%)，建议检查网络连接质量")
            
            # 基于流量数据生成建议
            for interface, history in self.metrics_history.items():
                if not history:
                    continue
                
                recent_metrics = list(history)[-10:]  # 最近10次测量
                total_errors = sum(m.errors_in + m.errors_out for m in recent_metrics)
                total_drops = sum(m.drops_in + m.drops_out for m in recent_metrics)
                
                if total_errors > 0:
                    recommendations.append(f"接口 {interface} 存在错误包，建议检查网络适配器")
                
                if total_drops > 0:
                    recommendations.append(f"接口 {interface} 存在丢包，建议检查网络负载或缓冲区设置")
            
            # 基于告警生成建议
            recent_critical_alerts = [
                a for a in self.alerts_history
                if a.severity == 'critical' and 
                (datetime.now() - a.timestamp).total_seconds() < 3600
            ]
            
            if len(recent_critical_alerts) > 5:
                recommendations.append("最近1小时内出现较多严重告警，建议进行全面的网络诊断")
            
            if not recommendations:
                recommendations.append("网络运行状态良好，建议继续监控")
        
        except Exception as e:
            self.logger.error(f"生成优化建议失败: {e}")
            recommendations.append("无法生成优化建议，请检查监控数据")
        
        return recommendations
    
    def export_data(self, output_file: str, data_type: str = 'all') -> bool:
        """导出监控数据"""
        try:
            export_data = {
                'export_info': {
                    'exported_at': datetime.now().isoformat(),
                    'data_type': data_type,
                    'version': 'W4 Network Monitor v1.0'
                }
            }
            
            if data_type in ['all', 'metrics']:
                export_data['network_metrics'] = {
                    interface: [asdict(metric) for metric in history]
                    for interface, history in self.metrics_history.items()
                }
            
            if data_type in ['all', 'performance']:
                export_data['network_performance'] = {
                    target: [asdict(perf) for perf in history]
                    for target, history in self.performance_history.items()
                }
            
            if data_type in ['all', 'alerts']:
                export_data['network_alerts'] = [asdict(alert) for alert in self.alerts_history]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"数据导出成功: {output_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"数据导出失败: {e}")
            return False
    
    def get_monitoring_status(self) -> Dict:
        """获取监控状态"""
        return {
            'is_monitoring': self.is_monitoring,
            'active_threads': len([t for t in self.monitor_threads.values() if t.is_alive()]),
            'interfaces': self.interfaces,
            'targets': self.config.get('targets', []),
            'data_points': {
                'metrics': sum(len(history) for history in self.metrics_history.values()),
                'performance': sum(len(history) for history in self.performance_history.values()),
                'alerts': len(self.alerts_history)
            },
            'uptime': time.time() - getattr(self, '_start_time', time.time())
        }


if __name__ == "__main__":
    # 示例用法
    monitor = NetworkMonitor()
    
    try:
        print("启动W4网络监控器...")
        monitor.start_monitoring()
        
        # 运行一段时间后生成报告
        time.sleep(30)
        
        print("生成网络报告...")
        report_file = monitor.generate_network_report()
        print(f"报告已生成: {report_file}")
        
        # 显示当前状态
        status = monitor.get_monitoring_status()
        print(f"监控状态: {status}")
        
        # 获取流量统计
        traffic_stats = monitor.get_network_traffic_stats()
        print(f"流量统计: {traffic_stats}")
        
        # 获取性能统计
        performance_stats = monitor.get_network_performance_stats()
        print(f"性能统计: {performance_stats}")
        
        # 获取告警
        alerts = monitor.get_network_alerts()
        print(f"告警数量: {len(alerts)}")
        
        input("按回车键停止监控...")
        
    except KeyboardInterrupt:
        print("\n正在停止监控...")
    finally:
        monitor.stop_monitoring()
        print("监控已停止")