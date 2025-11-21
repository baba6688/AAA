#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W3代理控制器包

这是一个功能完整的代理管理系统，提供代理管理、路由控制、流量控制等功能。
支持HTTP、HTTPS、SOCKS4、SOCKS5等多种代理协议。

主要组件:
- ProxyController: 主控制器
- ProxyManager: 代理管理器
- ProxyPool: 代理池
- TrafficController: 流量控制器
- ProxyRouter: 代理路由器
- ProxySecurityChecker: 安全检查器
- ProxyMonitor: 监控器
- ProxyHealthChecker: 健康检查器

数据类:
- ProxyConfig: 代理配置
- ProxyStats: 代理统计信息
- TrafficRecord: 流量记录

使用示例:
    from W3 import ProxyController
    
    controller = ProxyController()
    controller.start()
    
    controller.add_proxy(
        proxy_id="proxy1",
        host="127.0.0.1",
        port=8080,
        protocol="http"
    )
    
    response = controller.make_request("https://httpbin.org/ip")
    print(response.text)
    
    controller.stop()

版本: 1.0.0
作者: W3代理控制器开发团队
日期: 2025-11-06
"""

__version__ = "1.0.0"
__author__ = "W3代理控制器开发团队"
__email__ = "dev@w3proxy.com"
__license__ = "MIT"

# 导入主要类和函数
from .ProxyController import (
    ProxyController,
    ProxyConfig,
    ProxyStats,
    TrafficRecord,
    ProxyManager,
    ProxyPool,
    TrafficController,
    ProxyRouter,
    ProxySecurityChecker,
    ProxyMonitor,
    ProxyHealthChecker
)

# 导出的公共接口
__all__ = [
    "ProxyController",
    "ProxyConfig", 
    "ProxyStats",
    "TrafficRecord",
    "ProxyManager",
    "ProxyPool",
    "TrafficController",
    "ProxyRouter",
    "ProxySecurityChecker",
    "ProxyMonitor",
    "ProxyHealthChecker",
]

# 包级别的便捷函数
def create_controller():
    """
    创建一个新的代理控制器实例
    
    Returns:
        ProxyController: 代理控制器实例
    """
    return ProxyController()

def get_version():
    """
    获取版本信息
    
    Returns:
        str: 版本号
    """
    return __version__

def get_info():
    """
    获取包信息
    
    Returns:
        dict: 包含版本、作者等信息的字典
    """
    return {
        "name": "W3代理控制器",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": "功能完整的代理管理系统"
    }