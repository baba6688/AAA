#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W3代理控制器 - 主要实现
提供完整的代理管理、路由控制、流量控制等功能
"""

import asyncio
import json
import logging
import time
import threading
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import requests
from urllib.parse import urlparse
import socket
import ssl
import select
import queue
import weakref


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProxyConfig:
    """代理配置"""
    host: str
    port: int
    protocol: str  # http, https, socks4, socks5
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 30
    max_connections: int = 100
    priority: int = 1
    location: str = "unknown"
    is_active: bool = True
    last_check: Optional[datetime] = None
    response_time: float = 0.0
    success_rate: float = 1.0
    total_requests: int = 0
    failed_requests: int = 0
    bandwidth_limit: Optional[int] = None  # bytes per second
    whitelist: Optional[List[str]] = None
    blacklist: Optional[List[str]] = None


@dataclass
class ProxyStats:
    """代理统计信息"""
    proxy_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    avg_response_time: float = 0.0
    last_used: Optional[datetime] = None
    uptime: float = 0.0
    error_count: Dict[str, int] = None

    def __post_init__(self):
        if self.error_count is None:
            self.error_count = defaultdict(int)


@dataclass
class TrafficRecord:
    """流量记录"""
    timestamp: datetime
    proxy_id: str
    bytes_sent: int
    bytes_received: int
    response_time: float
    status_code: Optional[int] = None
    error_message: Optional[str] = None


class ProxyHealthChecker:
    """代理健康检查器"""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        
    def check_proxy_health(self, proxy_config: ProxyConfig) -> bool:
        """检查代理健康状态"""
        try:
            # 构建代理URL
            if proxy_config.protocol == 'http':
                proxy_url = f"http://{proxy_config.host}:{proxy_config.port}"
            elif proxy_config.protocol == 'https':
                proxy_url = f"https://{proxy_config.host}:{proxy_config.port}"
            elif proxy_config.protocol == 'socks4':
                proxy_url = f"socks4://{proxy_config.host}:{proxy_config.port}"
            elif proxy_config.protocol == 'socks5':
                proxy_url = f"socks5://{proxy_config.host}:{proxy_config.port}"
            else:
                return False
            
            # 设置认证
            if proxy_config.username and proxy_config.password:
                auth = (proxy_config.username, proxy_config.password)
            else:
                auth = None
            
            # 测试连接
            proxies = {proxy_config.protocol: proxy_url}
            start_time = time.time()
            
            response = requests.get(
                'http://httpbin.org/ip',
                proxies=proxies,
                auth=auth,
                timeout=proxy_config.timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                proxy_config.response_time = response_time
                proxy_config.last_check = datetime.now()
                return True
            else:
                return False
                
        except Exception as e:
            logger.warning(f"代理健康检查失败 {proxy_config.host}:{proxy_config.port}: {e}")
            return False
    
    def start_monitoring(self, proxy_manager):
        """开始监控"""
        self.running = True
        self.thread = threading.Thread(
            target=self._monitor_loop,
            args=(proxy_manager,),
            daemon=True
        )
        self.thread.start()
        logger.info("代理健康检查器已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("代理健康检查器已停止")
    
    def _monitor_loop(self, proxy_manager):
        """监控循环"""
        while self.running:
            try:
                for proxy_id, proxy_config in proxy_manager.proxies.items():
                    if proxy_config.is_active:
                        is_healthy = self.check_proxy_health(proxy_config)
                        if not is_healthy:
                            proxy_config.is_active = False
                            logger.warning(f"代理 {proxy_id} 健康检查失败，已禁用")
                        else:
                            logger.debug(f"代理 {proxy_id} 健康检查通过")
                
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"代理监控循环出错: {e}")
                time.sleep(self.check_interval)


class ProxyManager:
    """代理管理器"""
    
    def __init__(self):
        self.proxies: Dict[str, ProxyConfig] = {}
        self.proxy_stats: Dict[str, ProxyStats] = {}
        self.lock = threading.RLock()
        self.health_checker = ProxyHealthChecker()
        
    def add_proxy(self, proxy_id: str, config: ProxyConfig) -> bool:
        """添加代理"""
        with self.lock:
            self.proxies[proxy_id] = config
            self.proxy_stats[proxy_id] = ProxyStats(proxy_id=proxy_id)
            logger.info(f"添加代理: {proxy_id}")
            return True
    
    def remove_proxy(self, proxy_id: str) -> bool:
        """移除代理"""
        with self.lock:
            if proxy_id in self.proxies:
                del self.proxies[proxy_id]
                del self.proxy_stats[proxy_id]
                logger.info(f"移除代理: {proxy_id}")
                return True
            return False
    
    def get_proxy(self, proxy_id: str) -> Optional[ProxyConfig]:
        """获取代理配置"""
        with self.lock:
            return self.proxies.get(proxy_id)
    
    def list_proxies(self) -> Dict[str, ProxyConfig]:
        """列出所有代理"""
        with self.lock:
            return self.proxies.copy()
    
    def get_available_proxies(self, protocol: Optional[str] = None) -> List[str]:
        """获取可用代理列表"""
        with self.lock:
            available = []
            for proxy_id, config in self.proxies.items():
                if config.is_active and (protocol is None or config.protocol == protocol):
                    available.append(proxy_id)
            return available
    
    def update_proxy_stats(self, proxy_id: str, traffic_record: TrafficRecord):
        """更新代理统计"""
        with self.lock:
            if proxy_id in self.proxy_stats:
                stats = self.proxy_stats[proxy_id]
                stats.total_requests += 1
                stats.total_bytes_sent += traffic_record.bytes_sent
                stats.total_bytes_received += traffic_record.bytes_received
                
                if traffic_record.status_code and 200 <= traffic_record.status_code < 400:
                    stats.successful_requests += 1
                else:
                    stats.failed_requests += 1
                    if traffic_record.error_message:
                        stats.error_count[traffic_record.error_message] += 1
                
                # 更新平均响应时间
                if stats.total_requests > 0:
                    stats.avg_response_time = (
                        (stats.avg_response_time * (stats.total_requests - 1) + 
                         traffic_record.response_time) / stats.total_requests
                    )
                
                stats.last_used = traffic_record.timestamp
    
    def get_proxy_stats(self, proxy_id: str) -> Optional[ProxyStats]:
        """获取代理统计"""
        with self.lock:
            return self.proxy_stats.get(proxy_id)
    
    def start_health_check(self):
        """启动健康检查"""
        self.health_checker.start_monitoring(self)
    
    def stop_health_check(self):
        """停止健康检查"""
        self.health_checker.stop_monitoring()


class ProxyPool:
    """代理池"""
    
    def __init__(self, proxy_manager: ProxyManager):
        self.proxy_manager = proxy_manager
        self.current_index = 0
        self.lock = threading.RLock()
        
    def get_next_proxy(self, protocol: Optional[str] = None) -> Optional[str]:
        """获取下一个代理（轮询）"""
        with self.lock:
            available_proxies = self.proxy_manager.get_available_proxies(protocol)
            if not available_proxies:
                return None
            
            proxy_id = available_proxies[self.current_index % len(available_proxies)]
            self.current_index += 1
            return proxy_id
    
    def get_best_proxy(self, protocol: Optional[str] = None) -> Optional[str]:
        """获取最佳代理（基于成功率）"""
        with self.lock:
            available_proxies = self.proxy_manager.get_available_proxies(protocol)
            if not available_proxies:
                return None
            
            # 按优先级和成功率排序
            sorted_proxies = sorted(
                available_proxies,
                key=lambda pid: (
                    -self.proxy_manager.proxies[pid].priority,
                    -self.proxy_manager.proxy_stats.get(pid, ProxyStats(pid)).success_rate,
                    self.proxy_manager.proxies[pid].response_time
                )
            )
            return sorted_proxies[0] if sorted_proxies else None
    
    def get_random_proxy(self, protocol: Optional[str] = None) -> Optional[str]:
        """获取随机代理"""
        with self.lock:
            available_proxies = self.proxy_manager.get_available_proxies(protocol)
            return random.choice(available_proxies) if available_proxies else None


class TrafficController:
    """流量控制器"""
    
    def __init__(self):
        self.traffic_records: deque = deque(maxlen=10000)
        self.bandwidth_usage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.RLock()
        self.rate_limits: Dict[str, Tuple[int, float]] = {}  # (bytes, window_seconds)
        
    def add_traffic_record(self, record: TrafficRecord):
        """添加流量记录"""
        with self.lock:
            self.traffic_records.append(record)
            
            # 记录代理带宽使用情况
            proxy_bandwidth = self.bandwidth_usage[record.proxy_id]
            proxy_bandwidth.append((record.timestamp, record.bytes_sent + record.bytes_received))
    
    def check_rate_limit(self, proxy_id: str, bytes_to_send: int) -> bool:
        """检查速率限制"""
        with self.lock:
            if proxy_id not in self.rate_limits:
                return True
            
            limit_bytes, window_seconds = self.rate_limits[proxy_id]
            current_time = time.time()
            
            # 清理过期记录
            bandwidth_deque = self.bandwidth_usage[proxy_id]
            while bandwidth_deque and current_time - bandwidth_deque[0][0] > window_seconds:
                bandwidth_deque.popleft()
            
            # 计算当前窗口内的流量
            current_usage = sum(bytes_used for _, bytes_used in bandwidth_deque)
            
            return (current_usage + bytes_to_send) <= limit_bytes
    
    def set_rate_limit(self, proxy_id: str, bytes_limit: int, window_seconds: int):
        """设置速率限制"""
        with self.lock:
            self.rate_limits[proxy_id] = (bytes_limit, window_seconds)
    
    def get_traffic_stats(self, proxy_id: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """获取流量统计"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            if proxy_id:
                records = [r for r in self.traffic_records if r.proxy_id == proxy_id and r.timestamp >= cutoff_time]
            else:
                records = [r for r in self.traffic_records if r.timestamp >= cutoff_time]
            
            total_bytes = sum(r.bytes_sent + r.bytes_received for r in records)
            total_requests = len(records)
            avg_response_time = sum(r.response_time for r in records) / max(total_requests, 1)
            
            return {
                'total_bytes': total_bytes,
                'total_requests': total_requests,
                'avg_response_time': avg_response_time,
                'time_range_hours': hours
            }


class ProxyRouter:
    """代理路由器"""
    
    def __init__(self, proxy_pool: ProxyPool, traffic_controller: TrafficController):
        self.proxy_pool = proxy_pool
        self.traffic_controller = traffic_controller
        self.routes: Dict[str, str] = {}  # domain -> proxy_id
        self.default_proxy_strategy = 'best'  # best, random, round_robin
        self.lock = threading.RLock()
        
    def add_route(self, domain: str, proxy_id: str):
        """添加路由规则"""
        with self.lock:
            self.routes[domain] = proxy_id
            logger.info(f"添加路由规则: {domain} -> {proxy_id}")
    
    def remove_route(self, domain: str):
        """移除路由规则"""
        with self.lock:
            if domain in self.routes:
                del self.routes[domain]
                logger.info(f"移除路由规则: {domain}")
    
    def get_proxy_for_url(self, url: str) -> Optional[str]:
        """获取URL对应的代理"""
        with self.lock:
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower()
                
                # 检查特定域名路由
                if domain in self.routes:
                    return self.routes[domain]
                
                # 使用默认策略
                protocol = parsed_url.scheme.lower()
                if protocol == 'https':
                    protocol = 'https'
                elif protocol == 'http':
                    protocol = 'http'
                else:
                    protocol = None
                
                if self.default_proxy_strategy == 'best':
                    return self.proxy_pool.get_best_proxy(protocol)
                elif self.default_proxy_strategy == 'random':
                    return self.proxy_pool.get_random_proxy(protocol)
                else:  # round_robin
                    return self.proxy_pool.get_next_proxy(protocol)
                    
            except Exception as e:
                logger.error(f"获取代理路由失败: {e}")
                return None


class ProxySecurityChecker:
    """代理安全检查器"""
    
    def __init__(self):
        self.blocked_domains = set()
        self.suspicious_patterns = [
            r'.*\.onion$',
            r'.*\.i2p$',
            r'.*localhost.*',
            r'.*127\.0\.0\.1.*',
            r'.*0\.0\.0\.0.*'
        ]
        self.max_redirects = 5
        
    def is_safe_url(self, url: str) -> Tuple[bool, str]:
        """检查URL安全性"""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # 检查被阻止的域名
            if domain in self.blocked_domains:
                return False, "域名被阻止"
            
            # 检查可疑模式
            import re
            for pattern in self.suspicious_patterns:
                if re.match(pattern, domain):
                    return False, f"域名匹配可疑模式: {pattern}"
            
            # 检查协议
            if parsed_url.scheme not in ['http', 'https']:
                return False, f"不支持的协议: {parsed_url.scheme}"
            
            return True, "安全"
            
        except Exception as e:
            return False, f"URL解析错误: {e}"
    
    def is_safe_proxy(self, proxy_config: ProxyConfig) -> Tuple[bool, str]:
        """检查代理安全性"""
        try:
            # 检查端口范围
            if not (1 <= proxy_config.port <= 65535):
                return False, "端口号无效"
            
            # 检查主机名格式
            if proxy_config.host in ['localhost', '127.0.0.1', '0.0.0.0']:
                return False, "不允许使用本地地址"
            
            # 检查协议
            if proxy_config.protocol not in ['http', 'https', 'socks4', 'socks5']:
                return False, f"不支持的代理协议: {proxy_config.protocol}"
            
            return True, "安全"
            
        except Exception as e:
            return False, f"代理配置检查错误: {e}"


class ProxyMonitor:
    """代理监控器"""
    
    def __init__(self, proxy_manager: ProxyManager, traffic_controller: TrafficController):
        self.proxy_manager = proxy_manager
        self.traffic_controller = traffic_controller
        self.alerts: List[Dict[str, Any]] = []
        self.monitoring = False
        self.thread = None
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("代理监控器已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.thread:
            self.thread.join()
        logger.info("代理监控器已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                self._check_proxy_health()
                self._check_performance()
                self._check_traffic_anomalies()
                time.sleep(30)  # 每30秒检查一次
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(30)
    
    def _check_proxy_health(self):
        """检查代理健康状态"""
        for proxy_id, config in self.proxy_manager.proxies.items():
            if not config.is_active:
                self._add_alert('proxy_down', f"代理 {proxy_id} 离线", proxy_id)
            elif config.response_time > 10.0:  # 响应时间超过10秒
                self._add_alert('slow_proxy', f"代理 {proxy_id} 响应缓慢", proxy_id)
    
    def _check_performance(self):
        """检查性能指标"""
        for proxy_id, stats in self.proxy_manager.proxy_stats.items():
            if stats.total_requests > 0:
                success_rate = stats.successful_requests / stats.total_requests
                if success_rate < 0.8:  # 成功率低于80%
                    self._add_alert('low_success_rate', f"代理 {proxy_id} 成功率低", proxy_id)
    
    def _check_traffic_anomalies(self):
        """检查流量异常"""
        recent_stats = self.traffic_controller.get_traffic_stats(hours=1)
        if recent_stats['total_requests'] > 1000:  # 1小时内请求数过多
            self._add_alert('high_traffic', "流量异常高", None)
    
    def _add_alert(self, alert_type: str, message: str, proxy_id: Optional[str]):
        """添加告警"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'proxy_id': proxy_id
        }
        self.alerts.append(alert)
        
        # 保持最近1000条告警
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        logger.warning(f"告警: {message}")
    
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取告警信息"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert['timestamp'] >= cutoff_time]


class ProxyController:
    """W3代理控制器主类"""
    
    def __init__(self):
        self.proxy_manager = ProxyManager()
        self.proxy_pool = ProxyPool(self.proxy_manager)
        self.traffic_controller = TrafficController()
        self.router = ProxyRouter(self.proxy_pool, self.traffic_controller)
        self.security_checker = ProxySecurityChecker()
        self.monitor = ProxyMonitor(self.proxy_manager, self.traffic_controller)
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        
    def start(self):
        """启动代理控制器"""
        if self.running:
            return
        
        self.running = True
        self.proxy_manager.start_health_check()
        self.monitor.start_monitoring()
        logger.info("W3代理控制器已启动")
    
    def stop(self):
        """停止代理控制器"""
        if not self.running:
            return
        
        self.running = False
        self.proxy_manager.stop_health_check()
        self.monitor.stop_monitoring()
        self.executor.shutdown(wait=True)
        logger.info("W3代理控制器已停止")
    
    def add_proxy(self, proxy_id: str, host: str, port: int, protocol: str, 
                  username: Optional[str] = None, password: Optional[str] = None,
                  **kwargs) -> bool:
        """添加代理"""
        config = ProxyConfig(
            host=host,
            port=port,
            protocol=protocol,
            username=username,
            password=password,
            **kwargs
        )
        
        # 安全检查
        is_safe, reason = self.security_checker.is_safe_proxy(config)
        if not is_safe:
            logger.error(f"代理安全检查失败: {reason}")
            return False
        
        return self.proxy_manager.add_proxy(proxy_id, config)
    
    def remove_proxy(self, proxy_id: str) -> bool:
        """移除代理"""
        return self.proxy_manager.remove_proxy(proxy_id)
    
    def make_request(self, url: str, method: str = 'GET', **kwargs) -> requests.Response:
        """发起代理请求"""
        # 安全检查
        is_safe, reason = self.security_checker.is_safe_url(url)
        if not is_safe:
            raise ValueError(f"URL安全检查失败: {reason}")
        
        # 获取代理
        proxy_id = self.router.get_proxy_for_url(url)
        if not proxy_id:
            raise ValueError("没有可用的代理")
        
        proxy_config = self.proxy_manager.get_proxy(proxy_id)
        if not proxy_config:
            raise ValueError(f"代理配置不存在: {proxy_id}")
        
        # 构建代理URL
        if proxy_config.protocol == 'http':
            proxy_url = f"http://{proxy_config.host}:{proxy_config.port}"
        elif proxy_config.protocol == 'https':
            proxy_url = f"https://{proxy_config.host}:{proxy_config.port}"
        elif proxy_config.protocol == 'socks4':
            proxy_url = f"socks4://{proxy_config.host}:{proxy_config.port}"
        elif proxy_config.protocol == 'socks5':
            proxy_url = f"socks5://{proxy_config.host}:{proxy_config.port}"
        else:
            raise ValueError(f"不支持的代理协议: {proxy_config.protocol}")
        
        # 设置认证
        if proxy_config.username and proxy_config.password:
            auth = (proxy_config.username, proxy_config.password)
        else:
            auth = None
        
        # 设置代理
        proxies = {proxy_config.protocol: proxy_url}
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 发起请求
            response = requests.request(
                method=method,
                url=url,
                proxies=proxies,
                auth=auth,
                timeout=proxy_config.timeout,
                **kwargs
            )
            
            response_time = time.time() - start_time
            
            # 记录流量
            traffic_record = TrafficRecord(
                timestamp=datetime.now(),
                proxy_id=proxy_id,
                bytes_sent=len(response.request.body) if response.request.body else 0,
                bytes_received=len(response.content),
                response_time=response_time,
                status_code=response.status_code
            )
            
            self.traffic_controller.add_traffic_record(traffic_record)
            self.proxy_manager.update_proxy_stats(proxy_id, traffic_record)
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # 记录失败
            traffic_record = TrafficRecord(
                timestamp=datetime.now(),
                proxy_id=proxy_id,
                bytes_sent=0,
                bytes_received=0,
                response_time=response_time,
                error_message=str(e)
            )
            
            self.traffic_controller.add_traffic_record(traffic_record)
            self.proxy_manager.update_proxy_stats(proxy_id, traffic_record)
            
            raise
    
    def get_proxy_stats(self, proxy_id: Optional[str] = None) -> Dict[str, Any]:
        """获取代理统计"""
        if proxy_id:
            stats = self.proxy_manager.get_proxy_stats(proxy_id)
            return asdict(stats) if stats else {}
        else:
            return {pid: asdict(stats) for pid, stats in self.proxy_manager.proxy_stats.items()}
    
    def get_traffic_stats(self, proxy_id: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """获取流量统计"""
        return self.traffic_controller.get_traffic_stats(proxy_id, hours)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'running': self.running,
            'total_proxies': len(self.proxy_manager.proxies),
            'active_proxies': len(self.proxy_manager.get_available_proxies()),
            'total_requests': sum(stats.total_requests for stats in self.proxy_manager.proxy_stats.values()),
            'alerts_count': len(self.monitor.get_alerts(hours=1))
        }
    
    def add_route(self, domain: str, proxy_id: str):
        """添加路由规则"""
        self.router.add_route(domain, proxy_id)
    
    def remove_route(self, domain: str):
        """移除路由规则"""
        self.router.remove_route(domain)
    
    def set_rate_limit(self, proxy_id: str, bytes_limit: int, window_seconds: int):
        """设置速率限制"""
        self.traffic_controller.set_rate_limit(proxy_id, bytes_limit, window_seconds)
    
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取告警信息"""
        return self.monitor.get_alerts(hours)
    
    def export_config(self) -> Dict[str, Any]:
        """导出配置"""
        return {
            'proxies': {pid: asdict(config) for pid, config in self.proxy_manager.proxies.items()},
            'routes': self.router.routes,
            'rate_limits': self.traffic_controller.rate_limits
        }
    
    def import_config(self, config: Dict[str, Any]):
        """导入配置"""
        # 导入代理配置
        if 'proxies' in config:
            for proxy_id, proxy_data in config['proxies'].items():
                proxy_config = ProxyConfig(**proxy_data)
                self.proxy_manager.add_proxy(proxy_id, proxy_config)
        
        # 导入路由配置
        if 'routes' in config:
            for domain, proxy_id in config['routes'].items():
                self.router.add_route(domain, proxy_id)
        
        # 导入速率限制配置
        if 'rate_limits' in config:
            for proxy_id, (bytes_limit, window_seconds) in config['rate_limits'].items():
                self.traffic_controller.set_rate_limit(proxy_id, bytes_limit, window_seconds)


# 使用示例
if __name__ == "__main__":
    # 创建代理控制器实例
    controller = ProxyController()
    
    try:
        # 启动控制器
        controller.start()
        
        # 添加代理
        controller.add_proxy(
            proxy_id="proxy1",
            host="127.0.0.1",
            port=8080,
            protocol="http"
        )
        
        # 添加路由规则
        controller.add_route("example.com", "proxy1")
        
        # 发起请求
        response = controller.make_request("https://httpbin.org/ip")
        print(f"响应状态: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        # 获取统计信息
        stats = controller.get_proxy_stats()
        print(f"代理统计: {json.dumps(stats, indent=2, default=str)}")
        
    finally:
        # 停止控制器
        controller.stop()