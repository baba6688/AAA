"""
W2负载均衡器主要实现

功能包括：
- 多种负载均衡算法（轮询、最少连接、加权轮询等）
- 服务器管理和健康检查
- 会话保持和故障转移
- 性能监控和负载统计
- 配置管理
"""

import time
import threading
import logging
import json
import random
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# 可选导入requests，如果不可用则使用urllib
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class LoadBalanceAlgorithm(Enum):
    """负载均衡算法枚举"""
    ROUND_ROBIN = "round_robin"  # 轮询
    LEAST_CONNECTIONS = "least_connections"  # 最少连接
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # 加权轮询
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"  # 加权最少连接
    RANDOM = "random"  # 随机
    IP_HASH = "ip_hash"  # IP哈希


class ServerStatus(Enum):
    """服务器状态枚举"""
    HEALTHY = "healthy"  # 健康
    UNHEALTHY = "unhealthy"  # 不健康
    MAINTENANCE = "maintenance"  # 维护中
    DOWN = "down"  # 宕机


@dataclass
class Server:
    """服务器节点类"""
    id: str
    host: str
    port: int
    weight: int = 1
    status: ServerStatus = ServerStatus.HEALTHY
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    response_time_avg: float = 0.0
    last_health_check: Optional[datetime] = None
    health_check_url: Optional[str] = None
    health_check_interval: int = 30  # 秒
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def url(self) -> str:
        """获取服务器URL"""
        return f"http://{self.host}:{self.port}"
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    @property
    def is_healthy(self) -> bool:
        """检查服务器是否健康"""
        return self.status == ServerStatus.HEALTHY


@dataclass
class LoadBalancerConfig:
    """负载均衡器配置类"""
    algorithm: LoadBalanceAlgorithm = LoadBalanceAlgorithm.ROUND_ROBIN
    health_check_enabled: bool = True
    health_check_interval: int = 30
    health_check_timeout: int = 5
    health_check_path: str = "/health"
    max_retries: int = 3
    session_sticky_enabled: bool = False
    session_timeout: int = 3600  # 秒
    failover_enabled: bool = True
    performance_monitoring_enabled: bool = True
    max_connections_per_server: int = 1000
    connection_timeout: int = 10
    read_timeout: int = 30
    thread_pool_size: int = 10
    statistics_enabled: bool = True


class HealthChecker:
    """健康检查器"""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._thread = None
        
    def start(self):
        """启动健康检查"""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self._thread.start()
            self.logger.info("健康检查器已启动")
    
    def stop(self):
        """停止健康检查"""
        self._running = False
        if self._thread:
            self._thread.join()
        self.logger.info("健康检查器已停止")
    
    def check_server_health(self, server: Server) -> bool:
        """检查单个服务器健康状态"""
        try:
            if not server.health_check_url:
                server.health_check_url = f"{server.url}{self.config.health_check_path}"
            
            if HAS_REQUESTS:
                # 使用requests库
                response = requests.get(
                    server.health_check_url,
                    timeout=self.config.health_check_timeout
                )
                status_code = response.status_code
            else:
                # 使用urllib库
                req = urllib.request.Request(server.health_check_url)
                with urllib.request.urlopen(req, timeout=self.config.health_check_timeout) as response:
                    status_code = response.getcode()
            
            server.last_health_check = datetime.now()
            
            if status_code == 200:
                return True
            else:
                self.logger.warning(f"服务器 {server.id} 健康检查失败: HTTP {status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"服务器 {server.id} 健康检查异常: {e}")
            return False
    
    def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                # 这里需要从负载均衡器获取服务器列表
                # 实际实现中需要注入依赖
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"健康检查循环异常: {e}")


class SessionManager:
    """会话管理器"""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
    def get_session_server(self, session_id: str) -> Optional[str]:
        """获取会话对应的服务器ID"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if datetime.now() < session['expires_at']:
                return session['server_id']
            else:
                # 会话过期，清理
                del self.sessions[session_id]
        return None
    
    def set_session_server(self, session_id: str, server_id: str):
        """设置会话对应的服务器"""
        if self.config.session_sticky_enabled:
            expires_at = datetime.now() + timedelta(seconds=self.config.session_timeout)
            self.sessions[session_id] = {
                'server_id': server_id,
                'expires_at': expires_at,
                'created_at': datetime.now()
            }
    
    def clear_session(self, session_id: str):
        """清理会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]


class FailoverManager:
    """故障转移管理器"""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.failover_history: Dict[str, List[datetime]] = defaultdict(list)
        self.max_failover_attempts = 3
        
    def should_failover(self, server_id: str) -> bool:
        """判断是否应该进行故障转移"""
        if not self.config.failover_enabled:
            return False
            
        recent_failures = [
            failure for failure in self.failover_history[server_id]
            if datetime.now() - failure < timedelta(minutes=5)
        ]
        
        return len(recent_failures) >= self.max_failover_attempts
    
    def record_failure(self, server_id: str):
        """记录故障"""
        self.failover_history[server_id].append(datetime.now())
        
        # 清理过旧的记录
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.failover_history[server_id] = [
            failure for failure in self.failover_history[server_id]
            if failure > cutoff_time
        ]


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.request_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def record_request(self, server_id: str, response_time: float, success: bool):
        """记录请求信息"""
        if self.config.performance_monitoring_enabled:
            self.request_times[server_id].append({
                'timestamp': datetime.now(),
                'response_time': response_time,
                'success': success
            })
    
    def get_server_performance(self, server_id: str) -> Dict[str, Any]:
        """获取服务器性能统计"""
        if server_id not in self.request_times:
            return {
                'avg_response_time': 0.0,
                'requests_per_minute': 0,
                'success_rate': 1.0,
                'total_requests': 0
            }
        
        requests = list(self.request_times[server_id])
        total_requests = len(requests)
        
        if total_requests == 0:
            return {
                'avg_response_time': 0.0,
                'requests_per_minute': 0,
                'success_rate': 1.0,
                'total_requests': 0
            }
        
        # 计算平均响应时间
        response_times = [r['response_time'] for r in requests]
        avg_response_time = sum(response_times) / len(response_times)
        
        # 计算成功率
        successful_requests = sum(1 for r in requests if r['success'])
        success_rate = successful_requests / total_requests
        
        # 计算每分钟请求数
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        recent_requests = [
            r for r in requests 
            if r['timestamp'] > minute_ago
        ]
        requests_per_minute = len(recent_requests)
        
        return {
            'avg_response_time': avg_response_time,
            'requests_per_minute': requests_per_minute,
            'success_rate': success_rate,
            'total_requests': total_requests
        }


class LoadStatistics:
    """负载统计类"""
    
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.connection_counts: Dict[str, int] = defaultdict(int)
        self.start_time = datetime.now()
        
    def record_request(self, server_id: str):
        """记录请求"""
        if self.config.statistics_enabled:
            self.request_counts[server_id] += 1
    
    def record_connection(self, server_id: str, count: int):
        """记录连接数"""
        self.connection_counts[server_id] = count
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.config.statistics_enabled:
            return {}
        
        total_requests = sum(self.request_counts.values())
        uptime = datetime.now() - self.start_time
        
        stats = {
            'total_requests': total_requests,
            'uptime_seconds': uptime.total_seconds(),
            'requests_per_second': total_requests / max(uptime.total_seconds(), 1),
            'server_requests': dict(self.request_counts),
            'server_connections': dict(self.connection_counts)
        }
        
        return stats
    
    def reset(self):
        """重置统计"""
        self.request_counts.clear()
        self.connection_counts.clear()
        self.start_time = datetime.now()


class LoadBalancer:
    """负载均衡器主类"""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self.servers: Dict[str, Server] = {}
        self.algorithm_counter = 0
        self.weighted_servers = []
        
        # 初始化组件
        self.health_checker = HealthChecker(self.config)
        self.session_manager = SessionManager(self.config)
        self.failover_manager = FailoverManager(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.statistics = LoadStatistics(self.config)
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 启动健康检查
        if self.config.health_check_enabled:
            self.health_checker.start()
    
    def add_server(self, server: Server) -> bool:
        """添加服务器"""
        try:
            self.servers[server.id] = server
            self._update_weighted_servers()
            self.logger.info(f"添加服务器: {server.id} ({server.url})")
            return True
        except Exception as e:
            self.logger.error(f"添加服务器失败: {e}")
            return False
    
    def remove_server(self, server_id: str) -> bool:
        """移除服务器"""
        try:
            if server_id in self.servers:
                del self.servers[server_id]
                self._update_weighted_servers()
                self.logger.info(f"移除服务器: {server_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"移除服务器失败: {e}")
            return False
    
    def update_server(self, server_id: str, **kwargs) -> bool:
        """更新服务器信息"""
        try:
            if server_id in self.servers:
                server = self.servers[server_id]
                for key, value in kwargs.items():
                    if hasattr(server, key):
                        setattr(server, key, value)
                self._update_weighted_servers()
                self.logger.info(f"更新服务器: {server_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"更新服务器失败: {e}")
            return False
    
    def get_server(self, session_id: Optional[str] = None, client_ip: Optional[str] = None) -> Optional[Server]:
        """根据负载均衡算法获取服务器"""
        healthy_servers = [
            server for server in self.servers.values()
            if server.is_healthy and server.current_connections < self.config.max_connections_per_server
        ]
        
        if not healthy_servers:
            self.logger.warning("没有可用的健康服务器")
            return None
        
        # 会话粘性
        if self.config.session_sticky_enabled and session_id:
            sticky_server_id = self.session_manager.get_session_server(session_id)
            if sticky_server_id and sticky_server_id in self.servers:
                sticky_server = self.servers[sticky_server_id]
                if sticky_server.is_healthy:
                    return sticky_server
        
        # 根据算法选择服务器
        server = None
        
        if self.config.algorithm == LoadBalanceAlgorithm.ROUND_ROBIN:
            server = self._round_robin(healthy_servers)
        elif self.config.algorithm == LoadBalanceAlgorithm.LEAST_CONNECTIONS:
            server = self._least_connections(healthy_servers)
        elif self.config.algorithm == LoadBalanceAlgorithm.WEIGHTED_ROUND_ROBIN:
            server = self._weighted_round_robin(healthy_servers)
        elif self.config.algorithm == LoadBalanceAlgorithm.WEIGHTED_LEAST_CONNECTIONS:
            server = self._weighted_least_connections(healthy_servers)
        elif self.config.algorithm == LoadBalanceAlgorithm.RANDOM:
            server = self._random(healthy_servers)
        elif self.config.algorithm == LoadBalanceAlgorithm.IP_HASH:
            server = self._ip_hash(healthy_servers, client_ip)
        
        return server
    
    def _round_robin(self, servers: List[Server]) -> Server:
        """轮询算法"""
        if not servers:
            return None
        
        server = servers[self.algorithm_counter % len(servers)]
        self.algorithm_counter += 1
        return server
    
    def _least_connections(self, servers: List[Server]) -> Server:
        """最少连接算法"""
        return min(servers, key=lambda s: s.current_connections)
    
    def _weighted_round_robin(self, servers: List[Server]) -> Server:
        """加权轮询算法"""
        if not servers:
            return None
        
        # 重新计算权重
        current_weights = []
        for server in servers:
            weight = server.weight - server.current_connections
            current_weights.append(max(0, weight))
        
        total_weight = sum(current_weights)
        if total_weight == 0:
            return servers[0]
        
        # 随机选择
        random_weight = random.randint(1, total_weight)
        weight_sum = 0
        
        for i, weight in enumerate(current_weights):
            weight_sum += weight
            if random_weight <= weight_sum:
                return servers[i]
        
        return servers[0]
    
    def _weighted_least_connections(self, servers: List[Server]) -> Server:
        """加权最少连接算法"""
        if not servers:
            return None
        
        # 计算加权连接数
        weighted_connections = [
            (server.current_connections / server.weight, server)
            for server in servers
        ]
        
        return min(weighted_connections, key=lambda x: x[0])[1]
    
    def _random(self, servers: List[Server]) -> Server:
        """随机算法"""
        return random.choice(servers)
    
    def _ip_hash(self, servers: List[Server], client_ip: Optional[str]) -> Server:
        """IP哈希算法"""
        if not servers or not client_ip:
            return self._random(servers)
        
        hash_value = hash(client_ip)
        return servers[hash_value % len(servers)]
    
    def _update_weighted_servers(self):
        """更新加权服务器列表"""
        self.weighted_servers = []
        for server in self.servers.values():
            for _ in range(server.weight):
                self.weighted_servers.append(server)
    
    def forward_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """转发请求"""
        start_time = time.time()
        
        try:
            # 获取目标服务器
            session_id = request_data.get('session_id')
            client_ip = request_data.get('client_ip')
            server = self.get_server(session_id, client_ip)
            
            if not server:
                return {
                    'success': False,
                    'error': '没有可用的服务器',
                    'status_code': 503
                }
            
            # 增加连接数
            server.current_connections += 1
            
            try:
                # 记录会话
                if session_id:
                    self.session_manager.set_session_server(session_id, server.id)
                
                # 记录请求统计
                self.statistics.record_request(server.id)
                
                # 模拟请求处理（实际实现中会转发到真实服务器）
                response_data = self._process_request(server, request_data)
                
                # 更新服务器统计
                server.total_requests += 1
                if not response_data.get('success', True):
                    server.failed_requests += 1
                
                response_time = time.time() - start_time
                server.response_time_avg = (
                    (server.response_time_avg * (server.total_requests - 1) + response_time) 
                    / server.total_requests
                )
                
                # 记录性能监控
                self.performance_monitor.record_request(
                    server.id, response_time, response_data.get('success', True)
                )
                
                return response_data
                
            finally:
                # 减少连接数
                server.current_connections = max(0, server.current_connections - 1)
                self.statistics.record_connection(server.id, server.current_connections)
        
        except Exception as e:
            self.logger.error(f"请求处理异常: {e}")
            return {
                'success': False,
                'error': str(e),
                'status_code': 500
            }
    
    def _process_request(self, server: Server, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求（模拟实现）"""
        # 这里应该是实际的网络请求逻辑
        # 为了演示，我们模拟一个简单的响应
        time.sleep(0.1)  # 模拟网络延迟
        
        return {
            'success': True,
            'server_id': server.id,
            'server_url': server.url,
            'data': request_data.get('data', {}),
            'timestamp': datetime.now().isoformat(),
            'status_code': 200
        }
    
    def get_server_status(self, server_id: str) -> Optional[Dict[str, Any]]:
        """获取服务器状态"""
        if server_id not in self.servers:
            return None
        
        server = self.servers[server_id]
        performance = self.performance_monitor.get_server_performance(server_id)
        
        return {
            'id': server.id,
            'url': server.url,
            'status': server.status.value,
            'weight': server.weight,
            'current_connections': server.current_connections,
            'total_requests': server.total_requests,
            'failed_requests': server.failed_requests,
            'success_rate': server.success_rate,
            'response_time_avg': server.response_time_avg,
            'last_health_check': server.last_health_check.isoformat() if server.last_health_check else None,
            'performance': performance
        }
    
    def get_all_servers_status(self) -> List[Dict[str, Any]]:
        """获取所有服务器状态"""
        return [self.get_server_status(server_id) for server_id in self.servers.keys()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取负载均衡器统计信息"""
        return self.statistics.get_statistics()
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            'algorithm': self.config.algorithm.value,
            'health_check_enabled': self.config.health_check_enabled,
            'health_check_interval': self.config.health_check_interval,
            'session_sticky_enabled': self.config.session_sticky_enabled,
            'failover_enabled': self.config.failover_enabled,
            'performance_monitoring_enabled': self.config.performance_monitoring_enabled,
            'max_connections_per_server': self.config.max_connections_per_server,
            'thread_pool_size': self.config.thread_pool_size
        }
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.logger.info("配置已更新")
    
    def health_check_all(self) -> Dict[str, bool]:
        """对所有服务器进行健康检查"""
        results = {}
        for server_id, server in self.servers.items():
            results[server_id] = self.health_checker.check_server_health(server)
            if not results[server_id]:
                server.status = ServerStatus.UNHEALTHY
            else:
                server.status = ServerStatus.HEALTHY
        return results
    
    def shutdown(self):
        """关闭负载均衡器"""
        self.logger.info("正在关闭负载均衡器...")
        
        # 停止健康检查
        if self.config.health_check_enabled:
            self.health_checker.stop()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        self.logger.info("负载均衡器已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()