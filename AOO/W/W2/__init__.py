"""
W2负载均衡模块

本模块提供了完整的负载均衡解决方案，包括多种负载均衡算法、
服务器管理、健康检查、会话保持、故障转移、性能监控和统计等功能。

主要功能：
- 多种负载均衡算法（轮询、最少连接、加权轮询、随机、IP哈希等）
- 服务器节点管理和状态监控
- 自动健康检查和故障检测
- 会话保持和粘性会话
- 故障转移和自动恢复
- 实时性能监控
- 详细的负载统计

使用示例：
    from W.W2 import LoadBalancer, LoadBalancerConfig, LoadBalanceAlgorithm, Server, ServerStatus
    
    # 创建配置
    config = LoadBalancerConfig(
        algorithm=LoadBalanceAlgorithm.ROUND_ROBIN,
        health_check_enabled=True,
        session_sticky_enabled=True
    )
    
    # 创建负载均衡器
    lb = LoadBalancer(config)
    
    # 添加服务器
    server = Server(
        id="server1",
        host="192.168.1.100",
        port=8080,
        weight=2
    )
    lb.add_server(server)
    
    # 转发请求
    response = lb.forward_request({
        "client_ip": "192.168.1.50",
        "session_id": "session123",
        "data": {"message": "Hello"}
    })

版本：1.0.0
作者：W2团队
"""

# 从LoadBalancer模块导入所有主要类
from .LoadBalancer import (
    LoadBalanceAlgorithm,
    ServerStatus, 
    Server,
    LoadBalancerConfig,
    HealthChecker,
    SessionManager,
    FailoverManager,
    PerformanceMonitor,
    LoadStatistics,
    LoadBalancer
)

# 模块信息
__version__ = '1.0.0'
__author__ = 'W2团队'
__email__ = 'w2-team@example.com'
__license__ = 'MIT'

# 定义公共导出接口
__all__ = [
    # 主要类和配置
    'LoadBalancer',
    'LoadBalancerConfig', 
    'Server',
    
    # 枚举类型
    'LoadBalanceAlgorithm',
    'ServerStatus',
    
    # 组件类
    'HealthChecker',
    'SessionManager', 
    'FailoverManager',
    'PerformanceMonitor',
    'LoadStatistics'
]

# 便捷创建函数
def create_load_balancer(
    algorithm: LoadBalanceAlgorithm = LoadBalanceAlgorithm.ROUND_ROBIN,
    health_check_enabled: bool = True,
    session_sticky_enabled: bool = False,
    **kwargs
) -> LoadBalancer:
    """
    创建负载均衡器的便捷函数
    
    Args:
        algorithm: 负载均衡算法
        health_check_enabled: 是否启用健康检查
        session_sticky_enabled: 是否启用会话粘性
        **kwargs: 其他配置参数
        
    Returns:
        配置好的LoadBalancer实例
    """
    config = LoadBalancerConfig(
        algorithm=algorithm,
        health_check_enabled=health_check_enabled,
        session_sticky_enabled=session_sticky_enabled,
        **kwargs
    )
    return LoadBalancer(config)


def create_server(
    host: str,
    port: int,
    server_id: str = None,
    weight: int = 1,
    health_check_path: str = "/health"
) -> Server:
    """
    创建服务器的便捷函数
    
    Args:
        host: 服务器主机地址
        port: 端口号
        server_id: 服务器ID，如果为None则自动生成
        weight: 权重
        health_check_path: 健康检查路径
        
    Returns:
        配置好的Server实例
    """
    if server_id is None:
        server_id = f"{host}:{port}"
    
    return Server(
        id=server_id,
        host=host,
        port=port,
        weight=weight,
        health_check_url=f"http://{host}:{port}{health_check_path}"
    )


# 常量定义
DEFAULT_HEALTH_CHECK_PATH = "/health"
DEFAULT_SESSION_TIMEOUT = 3600
DEFAULT_MAX_CONNECTIONS = 1000
DEFAULT_HEALTH_CHECK_INTERVAL = 30
DEFAULT_THREAD_POOL_SIZE = 10

# 导出常量
__all__.extend([
    'create_load_balancer',
    'create_server',
    'DEFAULT_HEALTH_CHECK_PATH',
    'DEFAULT_SESSION_TIMEOUT', 
    'DEFAULT_MAX_CONNECTIONS',
    'DEFAULT_HEALTH_CHECK_INTERVAL',
    'DEFAULT_THREAD_POOL_SIZE'
])

# 模块初始化日志
import logging

logger = logging.getLogger(__name__)
logger.info(f"W2负载均衡模块已加载 - 版本 {__version__}")