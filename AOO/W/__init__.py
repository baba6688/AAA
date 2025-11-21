"""
W区 - 工作流模块：网络组件
Workflow Module - Network Components

模块描述：
W区为工作流模块提供全面的网络组件支持，包含网络连接管理、负载均衡、代理控制等
9个子模块，总计62个类和8,705行代码，提供完整的网络管理框架。

功能分类：
- W1: 网络连接管理器 (NetworkConnectionManager) - 15类连接管理
- W2: 负载均衡器 (LoadBalancer) - 9类负载均衡算法
- W3: 代理控制器 (ProxyController) - 10类代理管理
- W4: 网络监控器 (NetworkMonitor) - 4类网络监控
- W5: 防火墙管理器 (FirewallManager) - 3类防火墙控制
- W6: 网络安全 (NetworkSecurity) - 3类安全策略
- W7: 带宽管理器 (BandwidthManager) - 4类带宽控制
- W8: 网络诊断 (NetworkDiagnostics) - 4类诊断工具
- W9: 网络状态聚合器 (NetworkStatusAggregator) - 10类状态管理

版本：v1.0.0
最后更新：2025-11-14
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "MiniMax Agent"

# 主模块导入
from .W1.NetworkConnectionManager import (
    ConnectionState,
    ConnectionType,
    ConnectionConfig,
    ConnectionStats,
    Connection,
    TCPConnection,
    UDPConnection,
    HTTPConnection,
    SSLConnection,
    ConnectionPool,
    ConnectionMonitor,
    ConnectionRetryManager,
    ConnectionSecurityManager,
    NetworkConnectionManager,
    SecurityError
)

from .W2.LoadBalancer import (
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

from .W3.ProxyController import (
    ProxyConfig,
    ProxyStats,
    TrafficRecord,
    ProxyHealthChecker,
    ProxyManager,
    ProxyPool,
    TrafficController,
    ProxyRouter,
    ProxySecurityChecker,
    ProxyMonitor,
    ProxyController
)

from .W4.NetworkMonitor import (
    NetworkMetrics,
    NetworkPerformance,
    NetworkAlert,
    NetworkMonitor
)

from .W5.FirewallManager import (
    FirewallRule,
    SecurityEvent,
    FirewallManager
)

from .W6.NetworkSecurity import (
    SecurityEvent,
    SecurityReport,
    NetworkSecurity
)

from .W7.BandwidthManager import (
    BandwidthConfig,
    NetworkStats,
    QoSPolicy,
    BandwidthManager
)

from .W8.NetworkDiagnostics import (
    DiagnosticStatus,
    NetworkTestResult,
    DiagnosticReport,
    NetworkDiagnostics
)

from .W9.NetworkStatusAggregator import (
    NetworkModuleStatus,
    NetworkAlert,
    StatusCollector,
    DataAggregator,
    StatusAnalyzer,
    AlertManager,
    HistoryManager,
    ReportGenerator,
    Dashboard,
    NetworkStatusAggregator
)

# 导出配置
__all__ = [
    # W1 - 网络连接管理器 (15类)
    "ConnectionState", "ConnectionType", "ConnectionConfig", "ConnectionStats",
    "Connection", "TCPConnection", "UDPConnection", "HTTPConnection", "SSLConnection",
    "ConnectionPool", "ConnectionMonitor", "ConnectionRetryManager",
    "ConnectionSecurityManager", "NetworkConnectionManager", "SecurityError",
    
    # W2 - 负载均衡器 (9类)
    "LoadBalanceAlgorithm", "ServerStatus", "Server", "LoadBalancerConfig",
    "HealthChecker", "SessionManager", "FailoverManager", "PerformanceMonitor",
    "LoadStatistics", "LoadBalancer",
    
    # W3 - 代理控制器 (10类)
    "ProxyConfig", "ProxyStats", "TrafficRecord", "ProxyHealthChecker",
    "ProxyManager", "ProxyPool", "TrafficController", "ProxyRouter",
    "ProxySecurityChecker", "ProxyMonitor", "ProxyController",
    
    # W4 - 网络监控器 (4类)
    "NetworkMetrics", "NetworkPerformance", "NetworkAlert", "NetworkMonitor",
    
    # W5 - 防火墙管理器 (3类)
    "FirewallRule", "SecurityEvent", "FirewallManager",
    
    # W6 - 网络安全 (3类)
    "SecurityEvent", "SecurityReport", "NetworkSecurity",
    
    # W7 - 带宽管理器 (4类)
    "BandwidthConfig", "NetworkStats", "QoSPolicy", "BandwidthManager",
    
    # W8 - 网络诊断 (4类)
    "DiagnosticStatus", "NetworkTestResult", "DiagnosticReport", "NetworkDiagnostics",
    
    # W9 - 网络状态聚合器 (10类)
    "NetworkModuleStatus", "NetworkAlert", "StatusCollector", "DataAggregator",
    "StatusAnalyzer", "AlertManager", "HistoryManager", "ReportGenerator",
    "Dashboard", "NetworkStatusAggregator"
]

# 模块信息
MODULE_INFO = {
    "name": "Workflow Module - Network Components",
    "version": "1.0.0",
    "total_classes": 62,
    "total_lines": 8705,
    "sub_modules": {
        "W1": {"name": "Network Connection Manager", "classes": 15},
        "W2": {"name": "Load Balancer", "classes": 9},
        "W3": {"name": "Proxy Controller", "classes": 10},
        "W4": {"name": "Network Monitor", "classes": 4},
        "W5": {"name": "Firewall Manager", "classes": 3},
        "W6": {"name": "Network Security", "classes": 3},
        "W7": {"name": "Bandwidth Manager", "classes": 4},
        "W8": {"name": "Network Diagnostics", "classes": 4},
        "W9": {"name": "Network Status Aggregator", "classes": 10}
    }
}

print(f"W区 - 工作流模块已初始化，网络组件总数: {MODULE_INFO['total_classes']} 类")