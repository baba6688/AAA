#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O3网络优化器包

这个包提供了一个全面的网络优化解决方案，包含以下核心功能：
1. 网络连接池管理 - 支持连接复用、健康检查和超时管理
2. 请求批量处理 - 支持请求合并、批量发送和响应聚合
3. 网络超时和重试策略 - 智能重试机制和指数退避算法
4. 压缩和编码优化 - 数据压缩、传输编码和协议优化
5. 负载均衡和路由优化 - 多种负载均衡算法和智能路由
6. 网络监控和性能分析 - 实时监控延迟、带宽和连接状态
7. 异步网络优化处理 - 全异步支持高性能网络操作
8. 完整的错误处理和日志记录
9. 详细的文档字符串和使用示例

主要类和函数:
- NetworkOptimizer: 主要的网络优化器类
- AdvancedNetworkOptimizer: 高级网络优化器，包含熔断器、限流器等
- NetworkConfig: 网络配置类
- PerformanceBenchmark: 性能基准测试类

使用示例:
    import asyncio
    from O3 import NetworkOptimizer, NetworkConfig, LoadBalancingAlgorithm
    
    async def main():
        # 创建配置
        config = NetworkConfig(
            max_connections=100,
            load_balancing_algorithm=LoadBalancingAlgorithm.ROUND_ROBIN,
            enable_compression=True
        )
        
        # 创建优化器
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

Author: O3 Network Optimizer Team
Version: 1.0.0
Created: 2025-11-06
"""

from .NetworkOptimizer import (
    # 主要类和枚举
    NetworkOptimizer,
    AdvancedNetworkOptimizer,
    NetworkConfig,
    PerformanceBenchmark,
    
    # 枚举类
    CompressionType,
    LoadBalancingAlgorithm,
    RetryStrategy,
    HealthStatus,
    
    # 数据类
    RequestMetrics,
    ServerInfo,
    
    # 异常类
    NetworkOptimizerError,
    ConnectionError,
    TimeoutError,
    LoadBalancerError,
    
    # 工具函数
    create_optimizer,
    quick_request,
    retry_on_failure,
    measure_time,
    cache_result,
    
    # 高级组件
    CircuitBreaker,
    RateLimiter,
    DNSCache,
    SSLOptimizer,
    ProtocolOptimizer,
    ConnectionPoolAdvanced,
    RequestDeduplicator,
    PerformanceProfiler
)

__version__ = "1.0.0"
__author__ = "O3 Network Optimizer Team"
__email__ = "o3-optimizer@example.com"
__license__ = "MIT"

__all__ = [
    # 主要类
    "NetworkOptimizer",
    "AdvancedNetworkOptimizer", 
    "NetworkConfig",
    "PerformanceBenchmark",
    
    # 枚举
    "CompressionType",
    "LoadBalancingAlgorithm",
    "RetryStrategy",
    "HealthStatus",
    
    # 数据类
    "RequestMetrics",
    "ServerInfo",
    
    # 异常
    "NetworkOptimizerError",
    "ConnectionError", 
    "TimeoutError",
    "LoadBalancerError",
    
    # 工具函数
    "create_optimizer",
    "quick_request",
    "retry_on_failure",
    "measure_time",
    "cache_result",
    
    # 高级组件
    "CircuitBreaker",
    "RateLimiter",
    "DNSCache",
    "SSLOptimizer",
    "ProtocolOptimizer",
    "ConnectionPoolAdvanced",
    "RequestDeduplicator",
    "PerformanceProfiler"
]

# 包级别的配置
DEFAULT_CONFIG = NetworkConfig()

# 便捷函数
def create_simple_optimizer(
    servers: list = None,
    max_connections: int = 100,
    enable_compression: bool = True
) -> NetworkOptimizer:
    """
    创建简单的网络优化器
    
    Args:
        servers: 服务器列表，每个元素为(host, port, weight)
        max_connections: 最大连接数
        enable_compression: 是否启用压缩
        
    Returns:
        配置好的网络优化器实例
    """
    config = NetworkConfig(
        max_connections=max_connections,
        enable_compression=enable_compression
    )
    
    optimizer = NetworkOptimizer(config)
    
    if servers:
        for host, port, weight in servers:
            optimizer.add_server(host, port, weight)
    
    return optimizer


def get_version():
    """获取版本信息"""
    return __version__


def get_package_info():
    """获取包信息"""
    return {
        "name": "O3网络优化器",
        "version": __version__,
        "author": __author__,
        "description": "高性能网络优化解决方案",
        "license": __license__
    }