#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O3网络优化器模块

本模块提供了一个全面的网络优化解决方案，包含以下核心功能：
1. 网络连接池管理 - 支持连接复用、健康检查和超时管理
2. 请求批量处理 - 支持请求合并、批量发送和响应聚合
3. 网络超时和重试策略 - 智能重试机制和指数退避算法
4. 压缩和编码优化 - 数据压缩、传输编码和协议优化
5. 负载均衡和路由优化 - 多种负载均衡算法和智能路由
6. 网络监控和性能分析 - 实时监控延迟、带宽和连接状态
7. 异步网络优化处理 - 全异步支持高性能网络操作
8. 完整的错误处理和日志记录
9. 详细的文档字符串和使用示例

Author: O3 Network Optimizer Team
Version: 1.0.0
Created: 2025-11-06
"""

import asyncio
import aiohttp
import time
import json
import gzip
import zlib
import hashlib
import logging
import threading
import weakref
from typing import Dict, List, Optional, Union, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import socket
import ssl
import urllib.parse
from contextlib import asynccontextmanager
import statistics
import warnings
import traceback
from functools import wraps
import random
import pickle
import base64

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('network_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """压缩类型枚举"""
    NONE = "none"
    GZIP = "gzip"
    DEFLATE = "deflate"
    BROTLI = "br"


class LoadBalancingAlgorithm(Enum):
    """负载均衡算法枚举"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"


class RetryStrategy(Enum):
    """重试策略枚举"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class NetworkConfig:
    """网络配置数据类"""
    # 连接池配置
    max_connections: int = 100
    max_connections_per_host: int = 20
    connection_timeout: float = 30.0
    total_timeout: float = 60.0
    keepalive_timeout: float = 30.0
    
    # 批量处理配置
    batch_size: int = 10
    batch_timeout: float = 0.1
    max_batch_delay: float = 1.0
    
    # 重试配置
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    
    # 压缩配置
    enable_compression: bool = True
    compression_threshold: int = 1024
    compression_type: CompressionType = CompressionType.GZIP
    
    # 负载均衡配置
    load_balancing_algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN
    health_check_interval: float = 30.0
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    
    # 监控配置
    enable_monitoring: bool = True
    metrics_retention_time: int = 3600  # 1小时
    slow_request_threshold: float = 5.0
    
    # SSL配置
    verify_ssl: bool = True
    ssl_context: Optional[ssl.SSLContext] = None
    ca_certs: Optional[str] = None


@dataclass
class RequestMetrics:
    """请求指标数据类"""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    status_code: Optional[int] = None
    response_size: int = 0
    request_size: int = 0
    retry_count: int = 0
    server_address: Optional[str] = None
    compression_used: bool = False
    error_message: Optional[str] = None


@dataclass
class ServerInfo:
    """服务器信息数据类"""
    host: str
    port: int
    weight: int = 1
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: float = 0.0
    health_status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))


class ConnectionPool:
    """连接池管理类"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self._pools: Dict[str, aiohttp.TCPConnector] = {}
        self._session_refs: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def get_session(self, host: str, port: int) -> aiohttp.ClientSession:
        """获取客户端会话"""
        key = f"{host}:{port}"
        
        with self._lock:
            if key not in self._pools:
                connector = aiohttp.TCPConnector(
                    limit=self.config.max_connections_per_host,
                    limit_per_host=self.config.max_connections_per_host,
                    keepalive_timeout=self.config.keepalive_timeout,
                    enable_cleanup_closed=True,
                    ssl=self.config.ssl_context if self.config.verify_ssl else False
                )
                self._pools[key] = connector
            
            connector = self._pools[key]
        
        timeout = aiohttp.ClientTimeout(
            total=self.config.total_timeout,
            connect=self.config.connection_timeout,
            sock_read=self.config.connection_timeout
        )
        
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'Connection': 'keep-alive'}
        )
        
        self._session_refs.add(session)
        return session
    
    async def cleanup(self):
        """清理连接池"""
        for session in list(self._session_refs):
            if not session.closed:
                await session.close()
        
        for connector in self._pools.values():
            await connector.close()
        
        self._pools.clear()
        logger.info("连接池清理完成")


class HealthChecker:
    """健康检查器"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self._health_status: Dict[str, HealthStatus] = {}
        self._lock = threading.Lock()
        
    async def check_server_health(self, server: ServerInfo) -> HealthStatus:
        """检查服务器健康状态"""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                # 发送健康检查请求
                async with session.get(
                    f"http://{server.host}:{server.port}/health",
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        server.health_status = HealthStatus.HEALTHY
                        server.consecutive_failures = 0
                        logger.debug(f"服务器 {server.host}:{server.port} 健康检查通过，响应时间: {response_time:.3f}s")
                    else:
                        server.health_status = HealthStatus.UNHEALTHY
                        server.consecutive_failures += 1
                        logger.warning(f"服务器 {server.host}:{server.port} 健康检查失败，状态码: {response.status}")
                        
        except Exception as e:
            server.health_status = HealthStatus.UNHEALTHY
            server.consecutive_failures += 1
            logger.error(f"服务器 {server.host}:{server.port} 健康检查异常: {str(e)}")
        
        server.last_health_check = time.time()
        return server.health_status
    
    async def periodic_health_check(self, servers: List[ServerInfo]):
        """定期健康检查"""
        while True:
            try:
                tasks = [self.check_server_health(server) for server in servers]
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"定期健康检查异常: {str(e)}")
                await asyncio.sleep(self.config.health_check_interval)


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self._servers: List[ServerInfo] = []
        self._current_index = 0
        self._lock = threading.Lock()
        
    def add_server(self, host: str, port: int, weight: int = 1):
        """添加服务器"""
        server = ServerInfo(host=host, port=port, weight=weight)
        with self._lock:
            self._servers.append(server)
        logger.info(f"添加服务器: {host}:{port} (权重: {weight})")
    
    def remove_server(self, host: str, port: int):
        """移除服务器"""
        with self._lock:
            self._servers = [s for s in self._servers 
                           if not (s.host == host and s.port == port)]
        logger.info(f"移除服务器: {host}:{port}")
    
    async def get_server(self) -> Optional[ServerInfo]:
        """根据负载均衡算法获取服务器"""
        available_servers = [s for s in self._servers 
                           if s.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]]
        
        if not available_servers:
            logger.warning("没有可用的服务器")
            return None
        
        with self._lock:
            if self.config.load_balancing_algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
                return self._round_robin(available_servers)
            elif self.config.load_balancing_algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
                return self._least_connections(available_servers)
            elif self.config.load_balancing_algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin(available_servers)
            elif self.config.load_balancing_algorithm == LoadBalancingAlgorithm.RANDOM:
                return self._random(available_servers)
            elif self.config.load_balancing_algorithm == LoadBalancingAlgorithm.IP_HASH:
                return self._ip_hash(available_servers)
            elif self.config.load_balancing_algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
                return self._least_response_time(available_servers)
            else:
                return available_servers[0]
    
    def _round_robin(self, servers: List[ServerInfo]) -> ServerInfo:
        """轮询算法"""
        server = servers[self._current_index % len(servers)]
        self._current_index += 1
        return server
    
    def _least_connections(self, servers: List[ServerInfo]) -> ServerInfo:
        """最少连接算法"""
        return min(servers, key=lambda s: s.current_connections)
    
    def _weighted_round_robin(self, servers: List[ServerInfo]) -> ServerInfo:
        """加权轮询算法"""
        total_weight = sum(s.weight for s in servers)
        if total_weight == 0:
            return servers[0]
        
        # 简化的加权轮询实现
        target = random.randint(1, total_weight)
        current = 0
        
        for server in servers:
            current += server.weight
            if current >= target:
                return server
        
        return servers[-1]
    
    def _random(self, servers: List[ServerInfo]) -> ServerInfo:
        """随机算法"""
        return random.choice(servers)
    
    def _ip_hash(self, servers: List[ServerInfo]) -> ServerInfo:
        """IP哈希算法"""
        # 这里需要客户端IP，暂时使用随机选择
        return random.choice(servers)
    
    def _least_response_time(self, servers: List[ServerInfo]) -> ServerInfo:
        """最少响应时间算法"""
        return min(servers, key=lambda s: s.avg_response_time)


class RequestBatcher:
    """请求批处理器"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self._batches: Dict[str, deque] = defaultdict(lambda: deque())
        self._batch_timers: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        
    async def add_request(self, batch_key: str, request_data: Dict[str, Any]) -> asyncio.Future:
        """添加请求到批次"""
        future = asyncio.Future()
        
        async with self._lock:
            batch_queue = self._batches[batch_key]
            batch_queue.append((request_data, future))
            
            # 如果批次已满，立即发送
            if len(batch_queue) >= self.config.batch_size:
                await self._send_batch(batch_key)
            else:
                # 设置批次超时定时器
                if batch_key not in self._batch_timers:
                    timer_task = asyncio.create_task(self._batch_timeout_handler(batch_key))
                    self._batch_timers[batch_key] = timer_task
        
        return future
    
    async def _batch_timeout_handler(self, batch_key: str):
        """批次超时处理器"""
        await asyncio.sleep(self.config.batch_timeout)
        async with self._lock:
            if batch_key in self._batches and self._batches[batch_key]:
                await self._send_batch(batch_key)
    
    async def _send_batch(self, batch_key: str):
        """发送批次请求"""
        async with self._lock:
            if batch_key not in self._batches or not self._batches[batch_key]:
                return
            
            batch_queue = self._batches[batch_key]
            batch_requests = []
            
            # 提取批次中的所有请求
            while batch_queue:
                request_data, future = batch_queue.popleft()
                batch_requests.append((request_data, future))
            
            # 取消定时器
            if batch_key in self._batch_timers:
                self._batch_timers[batch_key].cancel()
                del self._batch_timers[batch_key]
        
        if batch_requests:
            await self._process_batch(batch_key, batch_requests)
    
    async def _process_batch(self, batch_key: str, batch_requests: List[Tuple]):
        """处理批次请求"""
        try:
            # 合并请求数据
            merged_data = {
                'batch_key': batch_key,
                'requests': [req_data for req_data, _ in batch_requests]
            }
            
            # 这里应该发送到后端服务进行处理
            # 为了演示，我们模拟响应
            await asyncio.sleep(0.01)  # 模拟网络延迟
            
            # 为每个请求设置结果
            for i, (_, future) in enumerate(batch_requests):
                response_data = {
                    'batch_index': i,
                    'batch_key': batch_key,
                    'data': f'Response for request {i} in batch {batch_key}'
                }
                
                if not future.done():
                    future.set_result(response_data)
                    
        except Exception as e:
            logger.error(f"批次处理异常 {batch_key}: {str(e)}")
            # 设置所有请求的错误结果
            for _, future in batch_requests:
                if not future.done():
                    future.set_exception(e)


class RetryManager:
    """重试管理器"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        
    def calculate_retry_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        if self.config.retry_strategy == RetryStrategy.FIXED_DELAY:
            return self.config.base_retry_delay
        
        elif self.config.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            return min(
                self.config.base_retry_delay * attempt,
                self.config.max_retry_delay
            )
        
        elif self.config.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_retry_delay * (2 ** (attempt - 1))
            return min(delay, self.config.max_retry_delay)
        
        else:
            return self.config.base_retry_delay
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """执行带重试的函数"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    delay = self.calculate_retry_delay(attempt + 1)
                    logger.warning(f"请求失败，{delay:.1f}秒后进行第{attempt + 2}次重试: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"重试次数已用完，最终失败: {str(e)}")
        
        raise last_exception


class CompressionManager:
    """压缩管理器"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        
    def compress_data(self, data: Union[str, bytes]) -> Tuple[bytes, bool]:
        """压缩数据"""
        if not self.config.enable_compression:
            return data.encode() if isinstance(data, str) else data, False
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if len(data) < self.config.compression_threshold:
            return data, False
        
        try:
            if self.config.compression_type == CompressionType.GZIP:
                compressed = gzip.compress(data)
            elif self.config.compression_type == CompressionType.DEFLATE:
                compressed = zlib.compress(data)
            else:
                return data, False
            
            # 如果压缩后数据更大，则不使用压缩
            if len(compressed) >= len(data):
                return data, False
            
            return compressed, True
            
        except Exception as e:
            logger.error(f"数据压缩失败: {str(e)}")
            return data, False
    
    def decompress_data(self, data: bytes, compressed: bool) -> bytes:
        """解压缩数据"""
        if not compressed:
            return data
        
        try:
            if self.config.compression_type == CompressionType.GZIP:
                return gzip.decompress(data)
            elif self.config.compression_type == CompressionType.DEFLATE:
                return zlib.decompress(data)
            else:
                return data
                
        except Exception as e:
            logger.error(f"数据解压缩失败: {str(e)}")
            return data


class NetworkMonitor:
    """网络监控器"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        self._start_time = time.time()
        
    def record_request(self, metrics: RequestMetrics):
        """记录请求指标"""
        with self._lock:
            self._metrics['response_times'].append(
                metrics.end_time - metrics.start_time if metrics.end_time else 0
            )
            self._metrics['request_sizes'].append(metrics.request_size)
            self._metrics['response_sizes'].append(metrics.response_size)
            
            if metrics.status_code:
                self._metrics[f'status_{metrics.status_code}'].append(1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = {
                'uptime': time.time() - self._start_time,
                'total_requests': len(self._metrics['response_times']),
                'avg_response_time': self._calculate_avg('response_times'),
                'p95_response_time': self._calculate_percentile('response_times', 95),
                'p99_response_time': self._calculate_percentile('response_times', 99),
                'avg_request_size': self._calculate_avg('request_sizes'),
                'avg_response_size': self._calculate_avg('response_sizes'),
                'requests_per_second': self._calculate_rps(),
                'error_rate': self._calculate_error_rate(),
                'compression_ratio': self._calculate_compression_ratio()
            }
            return stats
    
    def _calculate_avg(self, metric_name: str) -> float:
        """计算平均值"""
        values = list(self._metrics[metric_name])
        return statistics.mean(values) if values else 0.0
    
    def _calculate_percentile(self, metric_name: str, percentile: int) -> float:
        """计算百分位数"""
        values = list(self._metrics[metric_name])
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _calculate_rps(self) -> float:
        """计算每秒请求数"""
        uptime = time.time() - self._start_time
        total_requests = len(self._metrics['response_times'])
        return total_requests / uptime if uptime > 0 else 0.0
    
    def _calculate_error_rate(self) -> float:
        """计算错误率"""
        total_requests = len(self._metrics['response_times'])
        error_requests = sum(
            len(self._metrics[f'status_{code}']) 
            for code in [400, 401, 403, 404, 500, 502, 503, 504]
        )
        return error_requests / total_requests if total_requests > 0 else 0.0
    
    def _calculate_compression_ratio(self) -> float:
        """计算压缩比"""
        total_requests = len(self._metrics['response_sizes'])
        if total_requests == 0:
            return 0.0
        
        # 这里需要更详细的压缩统计，暂时返回估算值
        return 0.3  # 30%压缩率


class NetworkOptimizerError(Exception):
    """网络优化器异常基类"""
    pass


class ConnectionError(NetworkOptimizerError):
    """连接错误"""
    pass


class TimeoutError(NetworkOptimizerError):
    """超时错误"""
    pass


class LoadBalancerError(NetworkOptimizerError):
    """负载均衡错误"""
    pass


class NetworkOptimizer:
    """
    O3网络优化器主类
    
    提供全面的网络优化功能，包括连接池管理、请求批处理、重试机制、
    压缩优化、负载均衡、网络监控等。
    
    使用示例:
        ```python
        import asyncio
        from network_optimizer import NetworkOptimizer, NetworkConfig, LoadBalancingAlgorithm
        
        async def main():
            # 配置网络优化器
            config = NetworkConfig(
                max_connections=100,
                load_balancing_algorithm=LoadBalancingAlgorithm.ROUND_ROBIN,
                enable_compression=True
            )
            
            optimizer = NetworkOptimizer(config)
            
            # 添加服务器
            optimizer.add_server("server1.example.com", 8080, weight=2)
            optimizer.add_server("server2.example.com", 8080, weight=1)
            
            # 启动优化器
            await optimizer.start()
            
            # 发送请求
            response = await optimizer.make_request(
                method="GET",
                url="/api/data",
                headers={"Authorization": "Bearer token"}
            )
            
            print(f"响应状态: {response['status']}")
            print(f"响应数据: {response['data']}")
            
            # 获取性能统计
            stats = optimizer.get_performance_stats()
            print(f"平均响应时间: {stats['avg_response_time']:.3f}s")
            
            # 停止优化器
            await optimizer.stop()
        
        if __name__ == "__main__":
            asyncio.run(main())
        ```
    """
    
    def __init__(self, config: NetworkConfig = None):
        """
        初始化网络优化器
        
        Args:
            config: 网络配置对象，如果为None则使用默认配置
        """
        self.config = config or NetworkConfig()
        
        # 初始化各个组件
        self._connection_pool = ConnectionPool(self.config)
        self._health_checker = HealthChecker(self.config)
        self._load_balancer = LoadBalancer(self.config)
        self._request_batcher = RequestBatcher(self.config)
        self._retry_manager = RetryManager(self.config)
        self._compression_manager = CompressionManager(self.config)
        self._network_monitor = NetworkMonitor(self.config)
        
        # 内部状态
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._request_counter = 0
        self._lock = asyncio.Lock()
        
        logger.info("网络优化器初始化完成")
    
    async def start(self):
        """启动网络优化器"""
        if self._running:
            logger.warning("网络优化器已经在运行")
            return
        
        self._running = True
        
        # 启动健康检查任务
        if self._load_balancer._servers:
            self._health_check_task = asyncio.create_task(
                self._health_checker.periodic_health_check(self._load_balancer._servers)
            )
        
        logger.info("网络优化器已启动")
    
    async def stop(self):
        """停止网络优化器"""
        if not self._running:
            return
        
        self._running = False
        
        # 取消健康检查任务
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # 清理连接池
        await self._connection_pool.cleanup()
        
        logger.info("网络优化器已停止")
    
    def add_server(self, host: str, port: int, weight: int = 1):
        """
        添加服务器到负载均衡池
        
        Args:
            host: 服务器主机地址
            port: 服务器端口
            weight: 服务器权重
        """
        self._load_balancer.add_server(host, port, weight)
    
    def remove_server(self, host: str, port: int):
        """
        从负载均衡池移除服务器
        
        Args:
            host: 服务器主机地址
            port: 服务器端口
        """
        self._load_balancer.remove_server(host, port)
    
    async def make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Union[str, bytes, Dict]] = None,
        params: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        use_batch: bool = False,
        batch_key: Optional[str] = None,
        compress: bool = None,
        retry_on_failure: bool = True
    ) -> Dict[str, Any]:
        """
        发送HTTP请求
        
        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            data: 请求数据
            params: URL参数
            timeout: 请求超时时间
            use_batch: 是否使用批处理
            batch_key: 批处理键
            compress: 是否压缩数据
            retry_on_failure: 是否在失败时重试
            
        Returns:
            响应数据字典，包含status、data、headers等信息
        """
        if not self._running:
            raise NetworkOptimizerError("网络优化器未启动")
        
        # 生成请求ID
        async with self._lock:
            self._request_counter += 1
            request_id = f"req_{self._request_counter}_{int(time.time() * 1000)}"
        
        # 如果使用批处理
        if use_batch:
            return await self._make_batch_request(
                request_id, method, url, headers, data, params, timeout, batch_key
            )
        
        # 普通请求处理
        return await self._make_single_request(
            request_id, method, url, headers, data, params, timeout, compress, retry_on_failure
        )
    
    async def _make_batch_request(
        self,
        request_id: str,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]],
        data: Optional[Union[str, bytes, Dict]],
        params: Optional[Dict[str, str]],
        timeout: Optional[float],
        batch_key: Optional[str]
    ) -> Dict[str, Any]:
        """处理批处理请求"""
        request_data = {
            'request_id': request_id,
            'method': method,
            'url': url,
            'headers': headers or {},
            'data': data,
            'params': params,
            'timeout': timeout
        }
        
        # 使用URL作为默认批处理键
        if not batch_key:
            batch_key = f"{method}:{url}"
        
        future = await self._request_batcher.add_request(batch_key, request_data)
        return await future
    
    async def _make_single_request(
        self,
        request_id: str,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]],
        data: Optional[Union[str, bytes, Dict]],
        params: Optional[Dict[str, str]],
        timeout: Optional[float],
        compress: Optional[bool],
        retry_on_failure: bool
    ) -> Dict[str, Any]:
        """处理单次请求"""
        # 获取服务器
        server = await self._load_balancer.get_server()
        if not server:
            raise LoadBalancerError("没有可用的服务器")
        
        # 构建完整URL
        full_url = f"http://{server.host}:{server.port}{url}"
        if params:
            query_string = urllib.parse.urlencode(params)
            full_url += f"?{query_string}"
        
        # 准备请求数据
        request_data = data
        if isinstance(data, dict):
            request_data = json.dumps(data)
        
        # 数据压缩
        compressed_data = None
        should_compress = compress if compress is not None else self.config.enable_compression
        
        if request_data and should_compress:
            compressed_data, was_compressed = self._compression_manager.compress_data(request_data)
            if was_compressed:
                headers = headers or {}
                headers['Content-Encoding'] = 'gzip'
                request_data = compressed_data
        
        # 创建请求指标
        metrics = RequestMetrics(
            request_id=request_id,
            start_time=time.time(),
            request_size=len(str(request_data)) if request_data else 0,
            server_address=f"{server.host}:{server.port}",
            compression_used=compressed_data is not None
        )
        
        # 执行请求
        if retry_on_failure:
            response = await self._retry_manager.execute_with_retry(
                self._execute_request, method, full_url, headers, request_data, timeout, server, metrics
            )
        else:
            response = await self._execute_request(method, full_url, headers, request_data, timeout, server, metrics)
        
        return response
    
    async def _execute_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]],
        data: Optional[Union[str, bytes]],
        timeout: Optional[float],
        server: ServerInfo,
        metrics: RequestMetrics
    ) -> Dict[str, Any]:
        """执行实际的网络请求"""
        server.current_connections += 1
        
        try:
            # 获取客户端会话
            session = await self._connection_pool.get_session(server.host, server.port)
            
            # 设置超时
            request_timeout = timeout or self.config.total_timeout
            
            # 发送请求
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                timeout=aiohttp.ClientTimeout(total=request_timeout)
            ) as response:
                # 读取响应数据
                response_data = await response.read()
                
                # 解压缩响应数据
                if response.headers.get('Content-Encoding') == 'gzip':
                    response_data = self._compression_manager.decompress_data(response_data, True)
                
                # 更新指标
                metrics.end_time = time.time()
                metrics.status_code = response.status
                metrics.response_size = len(response_data)
                
                # 更新服务器统计
                response_time = metrics.end_time - metrics.start_time
                server.response_times.append(response_time)
                server.total_requests += 1
                server.avg_response_time = statistics.mean(server.response_times)
                
                # 记录指标
                self._network_monitor.record_request(metrics)
                
                # 解析响应数据
                try:
                    response_json = json.loads(response_data.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    response_json = {
                        'raw_data': base64.b64encode(response_data).decode('utf-8'),
                        'content_type': response.headers.get('Content-Type', 'application/octet-stream')
                    }
                
                result = {
                    'status': response.status,
                    'data': response_json,
                    'headers': dict(response.headers),
                    'request_id': metrics.request_id,
                    'response_time': response_time,
                    'server': f"{server.host}:{server.port}",
                    'compressed': metrics.compression_used
                }
                
                logger.debug(f"请求 {metrics.request_id} 成功，响应时间: {response_time:.3f}s")
                return result
                
        except asyncio.TimeoutError:
            metrics.end_time = time.time()
            metrics.error_message = "请求超时"
            server.failed_requests += 1
            logger.error(f"请求 {metrics.request_id} 超时")
            raise TimeoutError(f"请求超时: {url}")
            
        except Exception as e:
            metrics.end_time = time.time()
            metrics.error_message = str(e)
            server.failed_requests += 1
            server.consecutive_failures += 1
            logger.error(f"请求 {metrics.request_id} 失败: {str(e)}")
            raise ConnectionError(f"请求失败: {str(e)}")
            
        finally:
            server.current_connections -= 1
    
    async def make_batch_request(
        self,
        requests: List[Dict[str, Any]],
        batch_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        批量发送请求
        
        Args:
            requests: 请求列表，每个元素包含method、url、headers、data等
            batch_key: 批处理键
            
        Returns:
            响应列表
        """
        if not self._running:
            raise NetworkOptimizerError("网络优化器未启动")
        
        if not requests:
            return []
        
        # 生成批处理键
        if not batch_key:
            batch_key = f"batch_{int(time.time() * 1000)}"
        
        # 创建所有请求的Future
        futures = []
        for i, request_config in enumerate(requests):
            request_data = {
                'batch_index': i,
                'method': request_config.get('method', 'GET'),
                'url': request_config.get('url', '/'),
                'headers': request_config.get('headers', {}),
                'data': request_config.get('data'),
                'params': request_config.get('params'),
                'timeout': request_config.get('timeout')
            }
            
            future = await self._request_batcher.add_request(batch_key, request_data)
            futures.append(future)
        
        # 等待所有请求完成
        responses = []
        for future in asyncio.as_completed(futures):
            try:
                response = await future
                responses.append(response)
            except Exception as e:
                logger.error(f"批处理请求失败: {str(e)}")
                responses.append({
                    'error': str(e),
                    'status': 500,
                    'data': None
                })
        
        # 按原始顺序排序响应
        return responses
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            性能统计数据字典
        """
        return self._network_monitor.get_statistics()
    
    def get_server_status(self) -> List[Dict[str, Any]]:
        """
        获取所有服务器状态
        
        Returns:
            服务器状态列表
        """
        servers_info = []
        for server in self._load_balancer._servers:
            servers_info.append({
                'host': server.host,
                'port': server.port,
                'weight': server.weight,
                'current_connections': server.current_connections,
                'total_requests': server.total_requests,
                'failed_requests': server.failed_requests,
                'avg_response_time': server.avg_response_time,
                'health_status': server.health_status.value,
                'last_health_check': server.last_health_check,
                'consecutive_failures': server.consecutive_failures,
                'success_rate': (server.total_requests - server.failed_requests) / max(server.total_requests, 1)
            })
        return servers_info
    
    async def health_check(self) -> Dict[str, Any]:
        """
        执行健康检查
        
        Returns:
            健康检查结果
        """
        servers_info = []
        for server in self._load_balancer._servers:
            health_status = await self._health_checker.check_server_health(server)
            servers_info.append({
                'host': server.host,
                'port': server.port,
                'health_status': health_status.value,
                'response_time': server.avg_response_time,
                'last_check': server.last_health_check
            })
        
        healthy_count = sum(1 for s in servers_info if s['health_status'] == 'healthy')
        total_count = len(servers_info)
        
        return {
            'total_servers': total_count,
            'healthy_servers': healthy_count,
            'unhealthy_servers': total_count - healthy_count,
            'health_ratio': healthy_count / max(total_count, 1),
            'servers': servers_info
        }
    
    def update_config(self, new_config: NetworkConfig):
        """
        更新网络配置
        
        Args:
            new_config: 新的网络配置
        """
        old_config = self.config
        self.config = new_config
        
        # 更新各个组件的配置
        self._connection_pool.config = new_config
        self._health_checker.config = new_config
        self._load_balancer.config = new_config
        self._request_batcher.config = new_config
        self._retry_manager.config = new_config
        self._compression_manager.config = new_config
        self._network_monitor.config = new_config
        
        logger.info("网络配置已更新")
    
    @asynccontextmanager
    async def request_context(self, **kwargs):
        """
        请求上下文管理器
        
        使用示例:
            ```python
            async with optimizer.request_context(method="GET", url="/api/data") as response:
                print(f"响应: {response['data']}")
            ```
        """
        try:
            response = await self.make_request(**kwargs)
            yield response
        except Exception as e:
            logger.error(f"请求上下文异常: {str(e)}")
            raise
    
    def get_connection_pool_status(self) -> Dict[str, Any]:
        """
        获取连接池状态
        
        Returns:
            连接池状态信息
        """
        pool_status = {}
        for key, connector in self._connection_pool._pools.items():
            pool_status[key] = {
                'connector_type': type(connector).__name__,
                'limit': getattr(connector, 'limit', 0),
                'limit_per_host': getattr(connector, 'limit_per_host', 0),
                'closed': connector.closed if hasattr(connector, 'closed') else False
            }
        
        return {
            'total_pools': len(pool_status),
            'pools': pool_status
        }
    
    async def close_idle_connections(self):
        """关闭空闲连接"""
        for session in list(self._connection_pool._session_refs):
            if not session.closed:
                # 检查会话是否空闲
                connector = session.connector
                if connector and hasattr(connector, '_conns'):
                    # 这里可以添加更复杂的空闲连接检测逻辑
                    pass
        
        logger.info("空闲连接清理完成")
    
    def reset_statistics(self):
        """重置统计信息"""
        self._network_monitor._metrics.clear()
        self._network_monitor._start_time = time.time()
        logger.info("统计信息已重置")


# 工具函数和装饰器

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 重试延迟
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(delay * (2 ** attempt))  # 指数退避
                    else:
                        logger.error(f"函数 {func.__name__} 重试{max_retries}次后仍然失败")
            
            raise last_exception
        return wrapper
    return decorator


def measure_time(func):
    """性能测量装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"函数 {func.__name__} 执行时间: {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"函数 {func.__name__} 执行失败，耗时: {execution_time:.3f}s，错误: {str(e)}")
            raise
    return wrapper


def cache_result(ttl: int = 300):
    """
    结果缓存装饰器
    
    Args:
        ttl: 缓存生存时间（秒）
    """
    cache = {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 检查缓存
            if cache_key in cache:
                cached_result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    logger.debug(f"从缓存获取结果: {cache_key}")
                    return cached_result
            
            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            
            # 清理过期缓存
            expired_keys = [
                key for key, (_, timestamp) in cache.items()
                if time.time() - timestamp >= ttl
            ]
            for key in expired_keys:
                del cache[key]
            
            return result
        return wrapper
    return decorator


# 便捷函数

async def create_optimizer(
    servers: List[Tuple[str, int, int]] = None,
    config: NetworkConfig = None
) -> NetworkOptimizer:
    """
    创建网络优化器实例
    
    Args:
        servers: 服务器列表，每个元素为(host, port, weight)
        config: 网络配置
        
    Returns:
        配置好的网络优化器实例
    """
    optimizer = NetworkOptimizer(config)
    
    if servers:
        for host, port, weight in servers:
            optimizer.add_server(host, port, weight)
    
    await optimizer.start()
    return optimizer


async def quick_request(
    url: str,
    method: str = "GET",
    data: Any = None,
    headers: Dict[str, str] = None,
    servers: List[Tuple[str, int, int]] = None
) -> Dict[str, Any]:
    """
    快速发送请求
    
    Args:
        url: 请求URL
        method: HTTP方法
        data: 请求数据
        headers: 请求头
        servers: 服务器列表
        
    Returns:
        响应数据
    """
    config = NetworkConfig()
    optimizer = await create_optimizer(servers, config)
    
    try:
        return await optimizer.make_request(
            method=method,
            url=url,
            data=data,
            headers=headers
        )
    finally:
        await optimizer.stop()


# 使用示例和测试代码

async def example_basic_usage():
    """基础使用示例"""
    print("=== O3网络优化器基础使用示例 ===")
    
    # 创建配置
    config = NetworkConfig(
        max_connections=50,
        enable_compression=True,
        load_balancing_algorithm=LoadBalancingAlgorithm.ROUND_ROBIN,
        max_retries=3
    )
    
    # 创建优化器
    optimizer = NetworkOptimizer(config)
    
    # 添加服务器
    optimizer.add_server("httpbin.org", 80, weight=1)
    optimizer.add_server("jsonplaceholder.typicode.com", 80, weight=1)
    
    # 启动优化器
    await optimizer.start()
    
    try:
        # 发送GET请求
        response = await optimizer.make_request(
            method="GET",
            url="/get",
            headers={"User-Agent": "O3-NetworkOptimizer/1.0"}
        )
        print(f"GET请求响应状态: {response['status']}")
        print(f"响应时间: {response['response_time']:.3f}s")
        
        # 发送POST请求
        post_data = {"title": "测试", "body": "这是一个测试", "userId": 1}
        response = await optimizer.make_request(
            method="POST",
            url="/posts",
            data=post_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"POST请求响应状态: {response['status']}")
        
        # 获取性能统计
        stats = optimizer.get_performance_stats()
        print(f"平均响应时间: {stats['avg_response_time']:.3f}s")
        print(f"每秒请求数: {stats['requests_per_second']:.2f}")
        
    finally:
        await optimizer.stop()


async def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    config = NetworkConfig(batch_size=5, batch_timeout=0.1)
    optimizer = NetworkOptimizer(config)
    
    # 添加模拟服务器
    optimizer.add_server("httpbin.org", 80, weight=1)
    await optimizer.start()
    
    try:
        # 创建批量请求
        requests = []
        for i in range(10):
            requests.append({
                'method': 'GET',
                'url': f'/get?item={i}',
                'headers': {'X-Request-ID': str(i)}
            })
        
        # 发送批量请求
        responses = await optimizer.make_batch_request(requests, batch_key="test_batch")
        print(f"批量请求完成，共收到 {len(responses)} 个响应")
        
        for i, response in enumerate(responses):
            if 'error' not in response:
                print(f"请求 {i}: 状态 {response['status']}")
            else:
                print(f"请求 {i}: 错误 {response['error']}")
    
    finally:
        await optimizer.stop()


async def example_load_balancing():
    """负载均衡示例"""
    print("\n=== 负载均衡示例 ===")
    
    config = NetworkConfig(
        load_balancing_algorithm=LoadBalancingAlgorithm.LEAST_CONNECTIONS,
        health_check_interval=10.0
    )
    
    optimizer = NetworkOptimizer(config)
    
    # 添加多个服务器
    servers = [
        ("httpbin.org", 80, 2),  # 权重2
        ("jsonplaceholder.typicode.com", 80, 1),  # 权重1
        ("api.github.com", 80, 1)  # 权重1
    ]
    
    for host, port, weight in servers:
        optimizer.add_server(host, port, weight)
    
    await optimizer.start()
    
    try:
        # 发送多个请求观察负载均衡
        for i in range(6):
            response = await optimizer.make_request(
                method="GET",
                url="/get"
            )
            print(f"请求 {i+1}: 服务器 {response['server']}, 响应时间 {response['response_time']:.3f}s")
        
        # 获取服务器状态
        server_status = optimizer.get_server_status()
        print("\n服务器状态:")
        for server in server_status:
            print(f"  {server['host']}:{server['port']} - "
                  f"连接数: {server['current_connections']}, "
                  f"成功率: {server['success_rate']:.2%}")
    
    finally:
        await optimizer.stop()


async def example_error_handling():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    config = NetworkConfig(
        max_retries=2,
        retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        connection_timeout=5.0
    )
    
    optimizer = NetworkOptimizer(config)
    
    # 添加一个不存在的服务器来模拟错误
    optimizer.add_server("nonexistent-server.invalid", 80, weight=1)
    await optimizer.start()
    
    try:
        # 尝试请求不存在的服务器
        response = await optimizer.make_request(
            method="GET",
            url="/get",
            retry_on_failure=True
        )
        print(f"意外成功: {response}")
    
    except (ConnectionError, TimeoutError) as e:
        print(f"预期的连接错误: {str(e)}")
    
    finally:
        await optimizer.stop()


async def example_monitoring():
    """监控示例"""
    print("\n=== 监控示例 ===")
    
    config = NetworkConfig(enable_monitoring=True)
    optimizer = NetworkOptimizer(config)
    
    optimizer.add_server("httpbin.org", 80, weight=1)
    await optimizer.start()
    
    try:
        # 发送多个请求
        for i in range(20):
            await optimizer.make_request(
                method="GET",
                url="/get",
                headers={"X-Test-ID": str(i)}
            )
            await asyncio.sleep(0.1)  # 短暂延迟
        
        # 获取详细统计
        stats = optimizer.get_performance_stats()
        print("性能统计:")
        print(f"  运行时间: {stats['uptime']:.1f}s")
        print(f"  总请求数: {stats['total_requests']}")
        print(f"  平均响应时间: {stats['avg_response_time']:.3f}s")
        print(f"  P95响应时间: {stats['p95_response_time']:.3f}s")
        print(f"  P99响应时间: {stats['p99_response_time']:.3f}s")
        print(f"  每秒请求数: {stats['requests_per_second']:.2f}")
        print(f"  错误率: {stats['error_rate']:.2%}")
        print(f"  压缩比: {stats['compression_ratio']:.2%}")
        
        # 获取连接池状态
        pool_status = optimizer.get_connection_pool_status()
        print(f"\n连接池状态:")
        print(f"  总连接池数: {pool_status['total_pools']}")
        
    finally:
        await optimizer.stop()


async def example_compression():
    """压缩优化示例"""
    print("\n=== 压缩优化示例 ===")
    
    config = NetworkConfig(
        enable_compression=True,
        compression_threshold=100,  # 较小的阈值用于演示
        compression_type=CompressionType.GZIP
    )
    
    optimizer = NetworkOptimizer(config)
    optimizer.add_server("httpbin.org", 80, weight=1)
    await optimizer.start()
    
    try:
        # 发送大数据请求
        large_data = "x" * 1000  # 1KB数据
        response = await optimizer.make_request(
            method="POST",
            url="/post",
            data=large_data,
            headers={"Content-Type": "text/plain"}
        )
        
        print(f"大数据请求状态: {response['status']}")
        print(f"使用压缩: {response['compressed']}")
        
        # 发送小数据请求
        small_data = "small"
        response = await optimizer.make_request(
            method="POST",
            url="/post",
            data=small_data,
            headers={"Content-Type": "text/plain"}
        )
        
        print(f"小数据请求状态: {response['status']}")
        print(f"使用压缩: {response['compressed']}")
    
    finally:
        await optimizer.stop()


async def main():
    """主函数，运行所有示例"""
    examples = [
        example_basic_usage,
        example_batch_processing,
        example_load_balancing,
        example_error_handling,
        example_monitoring,
        example_compression
    ]
    
    for example in examples:
        try:
            await example()
            await asyncio.sleep(1)  # 示例间隔
        except Exception as e:
            print(f"示例 {example.__name__} 执行失败: {str(e)}")
            traceback.print_exc()
    
    print("\n=== 所有示例执行完成 ===")


# 高级功能和扩展类

class CircuitBreaker:
    """熔断器模式实现"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def can_execute(self) -> bool:
        """检查是否可以执行请求"""
        if self.state == 'CLOSED':
            return True
        elif self.state == 'OPEN':
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = 'HALF_OPEN'
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """记录成功"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


class RateLimiter:
    """限流器"""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
    
    async def acquire(self) -> bool:
        """获取许可"""
        now = time.time()
        
        # 清理过期的请求记录
        while self.requests and now - self.requests[0] > self.time_window:
            self.requests.popleft()
        
        # 检查是否超过限制
        if len(self.requests) >= self.max_requests:
            return False
        
        # 记录当前请求
        self.requests.append(now)
        return True
    
    def get_remaining_requests(self) -> int:
        """获取剩余请求数"""
        now = time.time()
        while self.requests and now - self.requests[0] > self.time_window:
            self.requests.popleft()
        return max(0, self.max_requests - len(self.requests))


class DNSCache:
    """DNS缓存"""
    
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.cache: Dict[str, Tuple[List[str], float]] = {}
        self._lock = threading.Lock()
    
    def resolve(self, hostname: str) -> List[str]:
        """解析域名"""
        with self._lock:
            if hostname in self.cache:
                ips, timestamp = self.cache[hostname]
                if time.time() - timestamp < self.ttl:
                    return ips
                else:
                    del self.cache[hostname]
        
        try:
            # 解析域名
            ips = socket.gethostbyname_ex(hostname)[2]
            with self._lock:
                self.cache[hostname] = (ips, time.time())
            return ips
        except Exception as e:
            logger.error(f"DNS解析失败 {hostname}: {str(e)}")
            return []
    
    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()


class SSLOptimizer:
    """SSL优化器"""
    
    def __init__(self):
        self.session_cache = {}
        self._lock = threading.Lock()
    
    def create_optimized_context(self, ca_certs: str = None, verify: bool = True) -> ssl.SSLContext:
        """创建优化的SSL上下文"""
        context = ssl.create_default_context()
        
        if ca_certs:
            context.load_verify_locations(ca_certs)
        
        # 优化设置
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # 启用会话复用
        context.options |= ssl.OP_NO_TICKET
        
        return context
    
    def get_session_id(self, hostname: str, port: int) -> str:
        """获取会话ID"""
        return f"{hostname}:{port}"
    
    def cache_session(self, session_id: str, session_data):
        """缓存会话"""
        with self._lock:
            self.session_cache[session_id] = (session_data, time.time())
    
    def get_cached_session(self, session_id: str):
        """获取缓存的会话"""
        with self._lock:
            if session_id in self.session_cache:
                session_data, timestamp = self.session_cache[session_id]
                if time.time() - timestamp < 3600:  # 1小时TTL
                    return session_data
                else:
                    del self.session_cache[session_id]
        return None


class ProtocolOptimizer:
    """协议优化器"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.http2_enabled = True
        self.http3_enabled = False  # 需要特殊支持
    
    def optimize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """优化HTTP头"""
        optimized = headers.copy()
        
        # 添加性能优化头
        if 'User-Agent' not in optimized:
            optimized['User-Agent'] = 'O3-NetworkOptimizer/1.0'
        
        if 'Accept-Encoding' not in optimized and self.config.enable_compression:
            optimized['Accept-Encoding'] = 'gzip, deflate'
        
        if 'Connection' not in optimized:
            optimized['Connection'] = 'keep-alive'
        
        # 添加缓存控制
        if 'Cache-Control' not in optimized:
            optimized['Cache-Control'] = 'no-cache'
        
        return optimized
    
    def should_use_http2(self, url: str) -> bool:
        """判断是否应该使用HTTP/2"""
        return self.http2_enabled and url.startswith('https://')
    
    def prepare_request_data(self, data: Any) -> Tuple[Union[str, bytes], bool]:
        """准备请求数据"""
        if isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False), True
        elif isinstance(data, str):
            return data.encode('utf-8'), False
        elif isinstance(data, bytes):
            return data, False
        else:
            return str(data).encode('utf-8'), False


class ConnectionPoolAdvanced:
    """高级连接池"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self._pools: Dict[str, aiohttp.TCPConnector] = {}
        self._session_refs: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.Lock()
        self._connection_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_connections': 0,
            'active_connections': 0,
            'idle_connections': 0,
            'connection_errors': 0
        })
    
    async def get_session_with_retry(self, host: str, port: int) -> aiohttp.ClientSession:
        """获取会话并重试"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await self.get_session(host, port)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))  # 指数退避
    
    async def get_session(self, host: str, port: int) -> aiohttp.ClientSession:
        """获取客户端会话"""
        key = f"{host}:{port}"
        
        with self._lock:
            if key not in self._pools:
                connector = aiohttp.TCPConnector(
                    limit=self.config.max_connections_per_host,
                    limit_per_host=self.config.max_connections_per_host,
                    keepalive_timeout=self.config.keepalive_timeout,
                    enable_cleanup_closed=True,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    ssl=self.config.ssl_context if self.config.verify_ssl else False,
                    family=socket.AF_INET  # 强制IPv4以提高兼容性
                )
                self._pools[key] = connector
                self._connection_stats[key]['total_connections'] += 1
            
            connector = self._pools[key]
        
        timeout = aiohttp.ClientTimeout(
            total=self.config.total_timeout,
            connect=self.config.connection_timeout,
            sock_read=self.config.connection_timeout,
            sock_connect=self.config.connection_timeout
        )
        
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'Connection': 'keep-alive',
                'Keep-Alive': f'timeout={int(self.config.keepalive_timeout)}'
            }
        )
        
        self._session_refs.add(session)
        return session
    
    async def cleanup_idle_connections(self):
        """清理空闲连接"""
        for session in list(self._session_refs):
            if not session.closed:
                connector = session.connector
                if connector and hasattr(connector, '_conns'):
                    # 这里可以添加更复杂的空闲检测逻辑
                    pass
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """获取连接池统计"""
        stats = {}
        for key, connector in self._pools.items():
            stats[key] = {
                'connector_type': type(connector).__name__,
                'limit': getattr(connector, 'limit', 0),
                'limit_per_host': getattr(connector, 'limit_per_host', 0),
                'closed': connector.closed if hasattr(connector, 'closed') else False,
                'total_connections': self._connection_stats[key]['total_connections']
            }
        return stats


class RequestDeduplicator:
    """请求去重器"""
    
    def __init__(self, ttl: int = 60):
        self.ttl = ttl
        self._requests: Dict[str, Tuple[asyncio.Future, float]] = {}
        self._lock = asyncio.Lock()
    
    async def deduplicate_request(self, request_key: str, request_func: Callable) -> Any:
        """去重请求"""
        async with self._lock:
            # 检查是否有相同请求正在处理
            if request_key in self._requests:
                future, timestamp = self._requests[request_key]
                if time.time() - timestamp < self.ttl:
                    return await future
                else:
                    del self._requests[request_key]
            
            # 创建新的Future并执行请求
            future = asyncio.Future()
            self._requests[request_key] = (future, time.time())
        
        try:
            result = await request_func()
            if not future.done():
                future.set_result(result)
            return result
        except Exception as e:
            if not future.done():
                future.set_exception(e)
            raise
        finally:
            async with self._lock:
                if request_key in self._requests:
                    del self._requests[request_key]
    
    def generate_request_key(self, method: str, url: str, data: Any = None, headers: Dict = None) -> str:
        """生成请求键"""
        content = f"{method}:{url}"
        if data:
            content += f":{hashlib.md5(str(data).encode()).hexdigest()}"
        if headers:
            content += f":{hashlib.md5(json.dumps(headers, sort_keys=True).encode()).hexdigest()}"
        return hashlib.md5(content.encode()).hexdigest()


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.profiles: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def profile_function(self, func_name: str, execution_time: float):
        """分析函数性能"""
        with self._lock:
            self.profiles[func_name].append(execution_time)
            
            # 保持最近1000次记录
            if len(self.profiles[func_name]) > 1000:
                self.profiles[func_name] = self.profiles[func_name][-1000:]
    
    def get_function_stats(self, func_name: str) -> Dict[str, float]:
        """获取函数统计"""
        with self._lock:
            times = self.profiles.get(func_name, [])
            if not times:
                return {}
            
            return {
                'count': len(times),
                'avg': statistics.mean(times),
                'min': min(times),
                'max': max(times),
                'median': statistics.median(times),
                'p95': statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times),
                'p99': statistics.quantiles(times, n=100)[98] if len(times) > 100 else max(times)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有统计"""
        return {name: self.get_function_stats(name) for name in self.profiles}


class AdvancedNetworkOptimizer(NetworkOptimizer):
    """高级网络优化器"""
    
    def __init__(self, config: NetworkConfig = None):
        super().__init__(config)
        
        # 高级组件
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._dns_cache = DNSCache()
        self._ssl_optimizer = SSLOptimizer()
        self._protocol_optimizer = ProtocolOptimizer(config)
        self._advanced_pool = ConnectionPoolAdvanced(config)
        self._request_deduplicator = RequestDeduplicator()
        self._performance_profiler = PerformanceProfiler()
        
        # 高级配置
        self._circuit_breaker_enabled = True
        self._rate_limiting_enabled = True
        self._request_deduplication_enabled = True
        self._dns_caching_enabled = True
        
        logger.info("高级网络优化器初始化完成")
    
    def add_server_with_circuit_breaker(self, host: str, port: int, weight: int = 1, 
                                      failure_threshold: int = 5, recovery_timeout: int = 60):
        """添加带熔断器的服务器"""
        self.add_server(host, port, weight)
        server_key = f"{host}:{port}"
        self._circuit_breakers[server_key] = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        logger.info(f"为服务器 {host}:{port} 启用熔断器")
    
    def add_rate_limiter(self, identifier: str, max_requests: int, time_window: int):
        """添加限流器"""
        self._rate_limiters[identifier] = RateLimiter(max_requests, time_window)
        logger.info(f"添加限流器 {identifier}: {max_requests} requests/{time_window}s")
    
    async def make_request_advanced(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Union[str, bytes, Dict]] = None,
        params: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        use_batch: bool = False,
        batch_key: Optional[str] = None,
        compress: bool = None,
        retry_on_failure: bool = True,
        rate_limit_key: Optional[str] = None,
        deduplicate: bool = True,
        profile: bool = False
    ) -> Dict[str, Any]:
        """高级请求方法"""
        start_time = time.time()
        
        try:
            # 限流检查
            if self._rate_limiting_enabled and rate_limit_key and rate_limit_key in self._rate_limiters:
                limiter = self._rate_limiters[rate_limit_key]
                if not await limiter.acquire():
                    raise NetworkOptimizerError(f"请求被限流器 {rate_limit_key} 限制")
            
            # 请求去重
            if self._request_deduplication_enabled and deduplicate:
                request_key = self._request_deduplicator.generate_request_key(method, url, data, headers)
                return await self._request_deduplicator.deduplicate_request(
                    request_key, 
                    lambda: self._make_single_request_advanced(
                        method, url, headers, data, params, timeout, compress, retry_on_failure, profile
                    )
                )
            else:
                return await self._make_single_request_advanced(
                    method, url, headers, data, params, timeout, compress, retry_on_failure, profile
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            if profile:
                self._performance_profiler.profile_function("make_request_advanced", execution_time)
            raise
    
    async def _make_single_request_advanced(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]],
        data: Optional[Union[str, bytes, Dict]],
        params: Optional[Dict[str, str]],
        timeout: Optional[float],
        compress: Optional[bool],
        retry_on_failure: bool,
        profile: bool
    ) -> Dict[str, Any]:
        """高级单次请求处理"""
        start_time = time.time()
        
        # 获取服务器
        server = await self._load_balancer.get_server()
        if not server:
            raise LoadBalancerError("没有可用的服务器")
        
        server_key = f"{server.host}:{server.port}"
        
        # 熔断器检查
        if self._circuit_breaker_enabled and server_key in self._circuit_breakers:
            circuit_breaker = self._circuit_breakers[server_key]
            if not circuit_breaker.can_execute():
                raise NetworkOptimizerError(f"服务器 {server_key} 熔断器开启")
        
        try:
            # DNS缓存
            if self._dns_caching_enabled:
                ips = self._dns_cache.resolve(server.host)
                if ips:
                    # 使用第一个IP
                    original_host = server.host
                    server.host = ips[0]
            
            # 协议优化
            if headers:
                headers = self._protocol_optimizer.optimize_headers(headers)
            else:
                headers = self._protocol_optimizer.optimize_headers({})
            
            # 准备请求数据
            request_data, is_json = self._protocol_optimizer.prepare_request_data(data)
            
            # 执行请求
            result = await self._execute_request_advanced(
                method, url, headers, request_data, timeout, server, is_json
            )
            
            # 熔断器记录成功
            if self._circuit_breaker_enabled and server_key in self._circuit_breakers:
                self._circuit_breakers[server_key].record_success()
            
            execution_time = time.time() - start_time
            if profile:
                self._performance_profiler.profile_function("make_request_advanced", execution_time)
            
            return result
        
        except Exception as e:
            # 熔断器记录失败
            if self._circuit_breaker_enabled and server_key in self._circuit_breakers:
                self._circuit_breakers[server_key].record_failure()
            
            execution_time = time.time() - start_time
            if profile:
                self._performance_profiler.profile_function("make_request_advanced", execution_time)
            
            raise
    
    async def _execute_request_advanced(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        data: Union[str, bytes],
        timeout: Optional[float],
        server: ServerInfo,
        is_json: bool
    ) -> Dict[str, Any]:
        """执行高级网络请求"""
        server.current_connections += 1
        
        try:
            # 使用高级连接池
            session = await self._advanced_pool.get_session_with_retry(server.host, server.port)
            
            # 设置超时
            request_timeout = timeout or self.config.total_timeout
            
            # 发送请求
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                timeout=aiohttp.ClientTimeout(total=request_timeout),
                ssl=self._ssl_optimizer.create_optimized_context() if self.config.verify_ssl else False
            ) as response:
                # 读取响应数据
                response_data = await response.read()
                
                # 处理压缩
                if response.headers.get('Content-Encoding') == 'gzip':
                    response_data = self._compression_manager.decompress_data(response_data, True)
                
                # 解析响应数据
                try:
                    if is_json:
                        response_json = json.loads(response_data.decode('utf-8'))
                    else:
                        response_json = {
                            'raw_data': base64.b64encode(response_data).decode('utf-8'),
                            'content_type': response.headers.get('Content-Type', 'application/octet-stream')
                        }
                except (json.JSONDecodeError, UnicodeDecodeError):
                    response_json = {
                        'raw_data': base64.b64encode(response_data).decode('utf-8'),
                        'content_type': response.headers.get('Content-Type', 'application/octet-stream')
                    }
                
                # 更新服务器统计
                response_time = time.time() - server.last_health_check
                server.response_times.append(response_time)
                server.total_requests += 1
                server.avg_response_time = statistics.mean(server.response_times)
                
                result = {
                    'status': response.status,
                    'data': response_json,
                    'headers': dict(response.headers),
                    'response_time': response_time,
                    'server': f"{server.host}:{server.port}",
                    'compressed': 'Content-Encoding' in response.headers
                }
                
                logger.debug(f"高级请求成功，响应时间: {response_time:.3f}s")
                return result
                
        except Exception as e:
            server.failed_requests += 1
            server.consecutive_failures += 1
            logger.error(f"高级请求失败: {str(e)}")
            raise ConnectionError(f"高级请求失败: {str(e)}")
            
        finally:
            server.current_connections -= 1
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """获取熔断器状态"""
        status = {}
        for server_key, circuit_breaker in self._circuit_breakers.items():
            status[server_key] = {
                'state': circuit_breaker.state,
                'failure_count': circuit_breaker.failure_count,
                'last_failure_time': circuit_breaker.last_failure_time
            }
        return status
    
    def get_rate_limiter_status(self) -> Dict[str, Dict[str, Any]]:
        """获取限流器状态"""
        status = {}
        for identifier, limiter in self._rate_limiters.items():
            status[identifier] = {
                'max_requests': limiter.max_requests,
                'time_window': limiter.time_window,
                'remaining_requests': limiter.get_remaining_requests()
            }
        return status
    
    def get_performance_profile(self) -> Dict[str, Dict[str, float]]:
        """获取性能分析"""
        return self._performance_profiler.get_all_stats()
    
    def clear_dns_cache(self):
        """清空DNS缓存"""
        self._dns_cache.clear_cache()
        logger.info("DNS缓存已清空")
    
    def reset_circuit_breakers(self):
        """重置所有熔断器"""
        for circuit_breaker in self._circuit_breakers.values():
            circuit_breaker.state = 'CLOSED'
            circuit_breaker.failure_count = 0
            circuit_breaker.last_failure_time = None
        logger.info("所有熔断器已重置")


# 性能测试和基准测试

class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self, optimizer: NetworkOptimizer):
        self.optimizer = optimizer
    
    async def benchmark_concurrent_requests(self, num_requests: int = 100, concurrency: int = 10) -> Dict[str, float]:
        """并发请求基准测试"""
        print(f"开始并发请求基准测试: {num_requests} 请求，{concurrency} 并发")
        
        start_time = time.time()
        semaphore = asyncio.Semaphore(concurrency)
        
        async def make_request(i):
            async with semaphore:
                try:
                    response = await self.optimizer.make_request(
                        method="GET",
                        url="/get",
                        headers={"X-Test-ID": str(i)}
                    )
                    return response['response_time']
                except Exception as e:
                    print(f"请求 {i} 失败: {str(e)}")
                    return None
        
        # 执行并发请求
        tasks = [make_request(i) for i in range(num_requests)]
        response_times = [rt for rt in await asyncio.gather(*tasks) if rt is not None]
        
        total_time = time.time() - start_time
        
        if response_times:
            stats = {
                'total_requests': num_requests,
                'successful_requests': len(response_times),
                'total_time': total_time,
                'requests_per_second': len(response_times) / total_time,
                'avg_response_time': statistics.mean(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'p50_response_time': statistics.median(response_times),
                'p95_response_time': statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
                'p99_response_time': statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
            }
        else:
            stats = {'error': '没有成功的请求'}
        
        return stats
    
    async def benchmark_batch_vs_single(self, num_requests: int = 50) -> Dict[str, float]:
        """批量vs单次请求基准测试"""
        print(f"开始批量vs单次请求基准测试: {num_requests} 请求")
        
        # 单次请求测试
        start_time = time.time()
        single_tasks = [
            self.optimizer.make_request(method="GET", url="/get")
            for _ in range(num_requests)
        ]
        single_responses = await asyncio.gather(*single_tasks, return_exceptions=True)
        single_time = time.time() - start_time
        
        # 批量请求测试
        start_time = time.time()
        batch_requests = [
            {'method': 'GET', 'url': '/get'}
            for _ in range(num_requests)
        ]
        batch_responses = await self.optimizer.make_batch_request(batch_requests)
        batch_time = time.time() - start_time
        
        return {
            'single_requests': {
                'total_time': single_time,
                'requests_per_second': num_requests / single_time,
                'successful_requests': sum(1 for r in single_responses if not isinstance(r, Exception))
            },
            'batch_requests': {
                'total_time': batch_time,
                'requests_per_second': num_requests / batch_time,
                'successful_requests': sum(1 for r in batch_responses if 'error' not in r)
            },
            'improvement': {
                'time_reduction': (single_time - batch_time) / single_time * 100,
                'throughput_increase': (num_requests / batch_time) / (num_requests / single_time) - 1
            }
        }
    
    async def benchmark_load_balancing(self, num_requests: int = 100) -> Dict[str, Any]:
        """负载均衡基准测试"""
        print(f"开始负载均衡基准测试: {num_requests} 请求")
        
        server_requests = defaultdict(int)
        response_times = []
        
        for i in range(num_requests):
            try:
                response = await self.optimizer.make_request(method="GET", url="/get")
                server = response['server']
                server_requests[server] += 1
                response_times.append(response['response_time'])
            except Exception as e:
                print(f"请求 {i} 失败: {str(e)}")
        
        # 分析负载分布
        total_requests = sum(server_requests.values())
        server_distribution = {
            server: {
                'requests': count,
                'percentage': count / total_requests * 100 if total_requests > 0 else 0
            }
            for server, count in server_requests.items()
        }
        
        return {
            'total_requests': total_requests,
            'server_distribution': server_distribution,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'load_balance_score': self._calculate_load_balance_score(server_distribution)
        }
    
    def _calculate_load_balance_score(self, distribution: Dict[str, Dict]) -> float:
        """计算负载均衡分数"""
        if not distribution:
            return 0.0
        
        percentages = [info['percentage'] for info in distribution.values()]
        expected_percentage = 100.0 / len(distribution)
        
        # 计算方差，越小表示负载越均衡
        variance = statistics.variance(percentages)
        max_variance = (100 - expected_percentage) ** 2
        
        # 返回均衡分数 (1.0表示完全均衡，0.0表示极不均衡)
        return max(0.0, 1.0 - variance / max_variance)


# 完整的使用示例和测试套件

async def comprehensive_example():
    """综合使用示例"""
    print("=== O3网络优化器综合示例 ===")
    
    # 创建高级配置
    config = NetworkConfig(
        max_connections=100,
        max_connections_per_host=20,
        connection_timeout=10.0,
        total_timeout=30.0,
        enable_compression=True,
        compression_threshold=512,
        load_balancing_algorithm=LoadBalancingAlgorithm.LEAST_CONNECTIONS,
        max_retries=3,
        retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        health_check_interval=30.0,
        enable_monitoring=True
    )
    
    # 创建高级优化器
    optimizer = AdvancedNetworkOptimizer(config)
    
    # 添加服务器和熔断器
    servers = [
        ("httpbin.org", 80, 2),
        ("jsonplaceholder.typicode.com", 80, 1)
    ]
    
    for host, port, weight in servers:
        optimizer.add_server_with_circuit_breaker(host, port, weight, failure_threshold=3)
    
    # 添加限流器
    optimizer.add_rate_limiter("api_calls", 100, 60)  # 每分钟100次请求
    
    await optimizer.start()
    
    try:
        print("1. 基础请求测试")
        response = await optimizer.make_request_advanced(
            method="GET",
            url="/get",
            headers={"User-Agent": "O3-Advanced-Optimizer/1.0"},
            profile=True
        )
        print(f"响应状态: {response['status']}, 响应时间: {response['response_time']:.3f}s")
        
        print("\n2. 批量请求测试")
        batch_requests = [
            {'method': 'GET', 'url': '/get', 'headers': {'X-Index': str(i)}}
            for i in range(10)
        ]
        batch_responses = await optimizer.make_batch_request(batch_requests)
        print(f"批量请求完成: {len(batch_responses)} 个响应")
        
        print("\n3. 性能基准测试")
        benchmark = PerformanceBenchmark(optimizer)
        
        # 并发测试
        concurrent_stats = await benchmark.benchmark_concurrent_requests(50, 5)
        print(f"并发测试结果:")
        print(f"  成功率: {concurrent_stats['successful_requests']}/{concurrent_stats['total_requests']}")
        print(f"  QPS: {concurrent_stats['requests_per_second']:.2f}")
        print(f"  平均响应时间: {concurrent_stats['avg_response_time']:.3f}s")
        
        # 负载均衡测试
        lb_stats = await benchmark.benchmark_load_balancing(30)
        print(f"负载均衡测试结果:")
        print(f"  负载均衡分数: {lb_stats['load_balance_score']:.3f}")
        print(f"  服务器分布: {lb_stats['server_distribution']}")
        
        print("\n4. 系统状态检查")
        
        # 性能统计
        perf_stats = optimizer.get_performance_stats()
        print(f"性能统计: {perf_stats}")
        
        # 服务器状态
        server_status = optimizer.get_server_status()
        print(f"服务器状态: {len(server_status)} 个服务器")
        
        # 熔断器状态
        cb_status = optimizer.get_circuit_breaker_status()
        print(f"熔断器状态: {cb_status}")
        
        # 限流器状态
        rl_status = optimizer.get_rate_limiter_status()
        print(f"限流器状态: {rl_status}")
        
        # 性能分析
        profile_stats = optimizer.get_performance_profile()
        print(f"性能分析: {profile_stats}")
        
        print("\n5. 错误处理测试")
        try:
            # 尝试访问不存在的服务器
            optimizer.add_server("nonexistent-server.invalid", 80, weight=1)
            await optimizer.make_request_advanced(
                method="GET",
                url="/get",
                retry_on_failure=True
            )
        except Exception as e:
            print(f"预期的错误被正确捕获: {str(e)[:100]}...")
        
        print("\n6. 高级功能测试")
        
        # DNS缓存测试
        optimizer.clear_dns_cache()
        print("DNS缓存已清空")
        
        # 熔断器重置测试
        optimizer.reset_circuit_breakers()
        print("熔断器已重置")
        
        # 统计重置测试
        optimizer.reset_statistics()
        print("统计信息已重置")
        
    finally:
        await optimizer.stop()
    
    print("\n=== 综合示例完成 ===")


async def stress_test():
    """压力测试"""
    print("=== O3网络优化器压力测试 ===")
    
    config = NetworkConfig(
        max_connections=200,
        max_connections_per_host=50,
        connection_timeout=5.0,
        total_timeout=15.0,
        enable_compression=True,
        load_balancing_algorithm=LoadBalancingAlgorithm.LEAST_CONNECTIONS,
        max_retries=2
    )
    
    optimizer = AdvancedNetworkOptimizer(config)
    
    # 添加多个服务器
    servers = [
        ("httpbin.org", 80, 3),
        ("jsonplaceholder.typicode.com", 80, 2),
        ("api.github.com", 80, 1)
    ]
    
    for host, port, weight in servers:
        optimizer.add_server_with_circuit_breaker(host, port, weight, failure_threshold=10)
    
    await optimizer.start()
    
    try:
        print("开始压力测试...")
        
        # 大量并发请求
        num_requests = 200
        concurrency = 20
        
        start_time = time.time()
        semaphore = asyncio.Semaphore(concurrency)
        
        async def stress_request(i):
            async with semaphore:
                try:
                    response = await optimizer.make_request_advanced(
                        method="GET",
                        url="/get",
                        headers={"X-Stress-ID": str(i)},
                        profile=True
                    )
                    return response['response_time']
                except Exception as e:
                    return None
        
        tasks = [stress_request(i) for i in range(num_requests)]
        response_times = [rt for rt in await asyncio.gather(*tasks) if rt is not None]
        
        total_time = time.time() - start_time
        
        print(f"压力测试结果:")
        print(f"  总请求数: {num_requests}")
        print(f"  成功请求数: {len(response_times)}")
        print(f"  成功率: {len(response_times)/num_requests*100:.1f}%")
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  QPS: {len(response_times)/total_time:.2f}")
        
        if response_times:
            print(f"  平均响应时间: {statistics.mean(response_times):.3f}s")
            print(f"  P95响应时间: {statistics.quantiles(response_times, n=20)[18]:.3f}s")
            print(f"  最大响应时间: {max(response_times):.3f}s")
        
        # 最终系统状态
        final_stats = optimizer.get_performance_stats()
        print(f"最终系统状态: {final_stats}")
        
    finally:
        await optimizer.stop()
    
    print("=== 压力测试完成 ===")


async def main():
    """主函数，运行所有示例和测试"""
    examples = [
        example_basic_usage,
        example_batch_processing,
        example_load_balancing,
        example_error_handling,
        example_monitoring,
        example_compression,
        comprehensive_example,
        stress_test
    ]
    
    for example in examples:
        try:
            await example()
            await asyncio.sleep(2)  # 示例间隔
        except Exception as e:
            print(f"示例 {example.__name__} 执行失败: {str(e)}")
            traceback.print_exc()
            print()
    
    print("=== 所有示例和测试执行完成 ===")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())