#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N2授权管理器包

一个完整的企业级授权管理系统，提供RBAC/ABAC权限控制、策略管理等核心功能。

主要模块:
- AuthorizationManager: 核心授权管理器类
- Subject: 主体类（用户、角色等）
- Resource: 资源类
- Action: 动作类
- Policy: 策略类
- Role: 角色类
- AuditLog: 审计日志类
- AuthorizationEngine: 授权引擎类

使用示例:
    from N2 import AuthorizationManager, Subject, Resource, Action
    
    auth_manager = AuthorizationManager()
    subject = Subject("user1", "user")
    resource = Resource("data1", ResourceType.DATA)
    action = Action(ActionType.READ)
    allowed, context = auth_manager.authorize(subject, resource, action)

版本: 1.0.0
日期: 2025-11-06
"""

from .AuthorizationManager import (
    PermissionEffect,
    ResourceType,
    ActionType,
    Subject,
    Resource,
    Action,
    Policy,
    Role,
    AuditLog,
    PermissionCache,
    RoleManager,
    PolicyManager,
    AuthorizationEngine,
    AuditManager,
    AuthorizationManager
)

__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__email__ = "security@example.com"
__license__ = "MIT"

# 包级别的公共接口
__all__ = [
    # 核心类
    "AuthorizationManager",
    
    # 基础类型
    "PermissionEffect",
    "ResourceType", 
    "ActionType",
    
    # 授权模型
    "Subject",
    "Resource",
    "Action",
    "Policy",
    "Role",
    
    # 管理和审计
    "AuditLog",
    "PermissionCache",
    "RoleManager",
    "PolicyManager",
    "AuditManager",
    
    # 引擎
    "AuthorizationEngine",
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
def create_auth_manager(config_file=None):
    """创建授权管理器实例的便捷函数
    
    Args:
        config_file: 配置文件路径，可选
        
    Returns:
        AuthorizationManager实例
    """
    return AuthorizationManager(config_file)

def create_subject(name, subject_type, **kwargs):
    """创建主体实例的便捷函数
    
    Args:
        name: 主体名称
        subject_type: 主体类型
        **kwargs: 其他主体属性
        
    Returns:
        Subject实例
    """
    return Subject(name=name, subject_type=subject_type, **kwargs)

def create_resource(name, resource_type, **kwargs):
    """创建资源实例的便捷函数
    
    Args:
        name: 资源名称
        resource_type: 资源类型
        **kwargs: 其他资源属性
        
    Returns:
        Resource实例
    """
    return Resource(name=name, resource_type=resource_type, **kwargs)

# 使用示例
EXAMPLE_USAGE = """
使用示例:

1. 基本授权:
   from N2 import AuthorizationManager, Subject, Resource, ActionType, ResourceType
   
   auth_manager = AuthorizationManager()
   subject = Subject("admin1", "user")
   resource = Resource("system_config", ResourceType.CONFIG)
   action = Action(ActionType.WRITE)
   allowed, context = auth_manager.authorize(subject, resource, action)
   if allowed:
       print("授权通过")

2. 角色管理:
   from N2 import Role, RoleManager
   
   role_manager = RoleManager()
   admin_role = Role("admin", "系统管理员", ["read", "write", "delete"])
   role_manager.add_role(admin_role)

3. 策略管理:
   from N2 import Policy, PolicyManager
   
   policy_manager = PolicyManager()
   policy = Policy(
       name="data_access_policy",
       resource_type=ResourceType.DATA,
       action_type=ActionType.READ,
       permission_effect=PermissionEffect.PERMIT
   )
   policy_manager.add_policy(policy)

4. 审计日志:
   from N2 import AuditManager, Subject, Resource, Action
   
   audit_manager = AuditManager()
   audit_entry = AuditLog(
       subject=Subject("user1", "user"),
       resource=Resource("file1", ResourceType.FILE),
       action=Action(ActionType.READ),
       result="ALLOW",
       timestamp=datetime.now()
   )
   audit_manager.log_access(audit_entry)
"""

if __name__ == "__main__":
    print("N2授权管理器包")
    print(f"版本: {__version__}")
    print(f"作者: {__author__}")
    print(f"许可证: {__license__}")
    print("\n" + "="*50)
    print(EXAMPLE_USAGE)