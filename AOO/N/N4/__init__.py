#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N4访问控制器包

一个完整的企业级访问控制系统，提供访问控制、网络监控、异常检测等核心功能。

主要模块:
- AccessController: 核心访问控制器类
- AccessRule: 访问规则类
- AccessEvent: 访问事件类
- NetworkAccessRule: 网络访问规则类
- AccessLevel: 访问级别枚举
- AccessType: 访问类型枚举
- RiskLevel: 风险级别枚举

使用示例:
    from N4 import AccessController, AccessRule, AccessLevel
    
    controller = AccessController()
    rule = AccessRule(
        subject="user1",
        resource="/admin/*",
        action=AccessLevel.FULL_ACCESS
    )
    controller.add_rule(rule)
    allowed = controller.check_access("user1", "/admin/dashboard")

版本: 1.0.0
日期: 2025-11-06
"""

from .AccessController import (
    AccessLevel,
    AccessType,
    RiskLevel,
    AccessRule,
    AccessEvent,
    NetworkAccessRule,
    AccessController
)

__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__email__ = "security@example.com"
__license__ = "MIT"

# 包级别的公共接口
__all__ = [
    # 核心类
    "AccessController",
    
    # 枚举类型
    "AccessLevel",
    "AccessType",
    "RiskLevel",
    
    # 规则和事件
    "AccessRule",
    "AccessEvent",
    "NetworkAccessRule",
]

# 包初始化信息
def get_version():
    """获取版本信息"""
    return __version__

def get_author():
    """获取作者信息"""
    return __author__

def get_license():
    """获取许可证信息"""
    return __license__

# 便捷函数
def create_controller(config_file=None):
    """创建访问控制器实例的便捷函数
    
    Args:
        config_file: 配置文件路径，可选
        
    Returns:
        AccessController实例
    """
    return AccessController(config_file)

def create_access_rule(subject, resource, access_level, **kwargs):
    """创建访问规则实例的便捷函数
    
    Args:
        subject: 访问主体
        resource: 资源路径
        access_level: 访问级别
        **kwargs: 其他规则属性
        
    Returns:
        AccessRule实例
    """
    return AccessRule(
        subject=subject,
        resource=resource,
        access_level=access_level,
        **kwargs
    )

def create_network_access_rule(ip_range, resource, **kwargs):
    """创建网络访问规则实例的便捷函数
    
    Args:
        ip_range: IP地址范围
        resource: 资源路径
        **kwargs: 其他规则属性
        
    Returns:
        NetworkAccessRule实例
    """
    return NetworkAccessRule(
        ip_range=ip_range,
        resource=resource,
        **kwargs
    )

# 使用示例
EXAMPLE_USAGE = """
使用示例:

1. 基础访问控制:
   from N4 import AccessController, AccessRule, AccessLevel
   
   controller = AccessController()
   rule = AccessRule(
       subject="admin",
       resource="/admin/*",
       action=AccessLevel.FULL_ACCESS
   )
   controller.add_rule(rule)
   allowed = controller.check_access("admin", "/admin/dashboard")
   print(f"访问权限: {'允许' if allowed else '拒绝'}")

2. 网络访问规则:
   from N4 import NetworkAccessRule, AccessType
   
   network_rule = NetworkAccessRule(
       ip_range="192.168.1.0/24",
       resource="/api/*",
       action=AccessType.WRITE,
       allowed=True
   )
   controller.add_network_rule(network_rule)

3. 访问事件监控:
   from N4 import AccessEvent, AccessController
   
   controller = AccessController()
   event = AccessEvent(
       user_id="user123",
       resource="/confidential/data",
       action=AccessLevel.READ,
       result="ALLOW",
       ip_address="192.168.1.100",
       timestamp=datetime.now()
   )
   controller.log_access_event(event)

4. 风险评估:
   from N4 import AccessController, RiskLevel
   
   controller = AccessController()
   risk_level = controller.assess_risk("user123", "/sensitive/data", AccessLevel.WRITE)
   if risk_level == RiskLevel.HIGH:
       print("高风险访问，需要额外验证")

5. 批量规则管理:
   from N4 import AccessController, AccessRule, AccessLevel
   
   controller = AccessController()
   rules = [
       AccessRule("user1", "/data/report", AccessLevel.READ),
       AccessRule("user1", "/data/admin", AccessLevel.FULL_ACCESS),
       AccessRule("user2", "/data/report", AccessLevel.READ)
   ]
   controller.add_rules(rules)
"""

if __name__ == "__main__":
    print("N4访问控制器包")
    print(f"版本: {__version__}")
    print(f"作者: {__author__}")
    print(f"许可证: {__license__}")
    print("\n" + "="*50)
    print(EXAMPLE_USAGE)